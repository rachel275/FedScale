# -*- coding: utf-8 -*-
import collections
import gc
import json
import os
import pickle
import random
import time
import uuid
from argparse import Namespace

import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.channels.channel_context import ClientConnections
#from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.execution import gemm_trace
from fedscale.cloud.fllibs import *
import fedscale.cloud.fllibs as fllibs
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from fedscale.cloud.execution.communication_metrics import (
    append_communication_record,
    raw_tensor_bytes,
)
from fedscale.cloud.channels.chunked_transfer import (
    iter_chunks,
    reassemble_chunks,
)

class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # initiate the executor log path, and executor ips
        logger.initiate_client_setting()

       # self.model_adapter = self.get_client_trainer(args).get_model_adapter(
       #     init_model()
       # )

        self.args = args
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)
        gemm_trace.configure(
            executor_id=self.executor_id,
        )
        
        self.model_adapter = self.get_client_trainer(args).get_model_adapter(
                init_model()
        )

        traced_modules = gemm_trace.attach(
            self.model_adapter.get_model()
        )

        logging.info(
            "GEMM tracer attached to %d Linear modules; output=%s",
            traced_modules,
            gemm_trace.trace_path(),
        )


        self.communication_log_path = os.path.join(
            self.args.log_path,
            f"communication-executor-{self.executor_id}.jsonl",
        )
        self.client_update_log_path = os.path.join(
            self.args.log_path,
            f"client-update-executor-{self.executor_id}.jsonl",
        )
        # ======== model and data ========
        self.training_sets = self.test_dataset = None

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        if args.wandb_token != "":
            os.environ["WANDB_API_KEY"] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=f"fedscale-{args.job_name}",
                    name=f"executor{args.this_rank}-{args.time_stamp}",
                    group=f"{args.time_stamp}",
                )
            else:
                logging.error("Warning: wandb has already been initialized")

        else:
            self.wandb = None
        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment"""
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        """Set up grpc connection"""
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)"""
        pass

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == "nlp":
            self.collate_fn = collate
        elif self.args.task == "voice":
            self.collate_fn = voice_collate_fn
        # load data partitionxr (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class
        )
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants,
            data_map_file=self.args.data_map_file,
        )

        testing_sets = DataPartitioner(
            data=test_dataset,
            args=self.args,
            numOfClass=self.args.num_class,
            isTest=True,
        )
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message."""
        self.setup_env()
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request, rpc_duration_s=None):
        """Add new events to worker queues.

        rpc_duration_s is the application-observed duration of the RPC that
        returned this response. It includes gRPC/control-plane overhead and,
        depending on the call, may include server processing or polling wait.
        """
        self.event_queue.append((request, rpc_duration_s))

    @staticmethod
    def _append_jsonl(path, record):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.

        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): TorchClient responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def download_streamed_payload(self, request_data):
        """Download and reassemble a large payload referenced by request_data."""
        transfer_info = self.deserialize_response(request_data)

        transfer_id = transfer_info["transfer_id"]
        expected_serialized_bytes = transfer_info.get("serialized_bytes")

        download_start = time.perf_counter()

        chunk_stream = (
            self.aggregator_communicator.stub.DOWNLOAD_DATA(
                job_api_pb2.TransferRequest(
                    transfer_id=transfer_id,
                    executor_id=self.executor_id,
                )
            )
        )

        serialized_payload = reassemble_chunks(chunk_stream)

        download_duration_s = time.perf_counter() - download_start

        if (
            expected_serialized_bytes is not None
            and len(serialized_payload) != expected_serialized_bytes
        ):
            raise RuntimeError(
                "Downloaded payload size mismatch: "
                f"expected {expected_serialized_bytes} bytes, "
                f"got {len(serialized_payload)} bytes"
            )

        throughput_mbps = (
            len(serialized_payload)
            * 8
            / download_duration_s
            / 1_000_000
            if download_duration_s > 0
            else None
        )

        return (
            serialized_payload,
            download_duration_s,
            throughput_mbps,
        )

    def set_received_weights(self, weights):
        """Apply full-model or LoRA adapter weights received from aggregator."""
        if getattr(self.args, "method", "full") == "lora":
            self.model_adapter.set_lora_weights(weights)
        else:
            self.model_adapter.set_weights(
                weights,
                is_aggregator=False,
            )

    def UpdateModel(self, model_weights):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config

        """
        self.round += 1
        self.set_received_weights(model_weights) #, is_aggregator=False)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config = config["client_id"], config["task_config"]

        if "model" not in config or not config["model"]:
            raise "The 'model' object must be a non-null value in the training config."
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            client_id=client_id, conf=client_conf, model=config["model"]
        )

        # Report execution completion meta information.
        rpc_start = time.perf_counter()
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=str(client_id),
                executor_id=self.executor_id,
                event=commons.CLIENT_TRAIN,
                status=True,
                msg=None,
                meta_result=None,
                data_result=None,
            )
        )
        rpc_duration_s = time.perf_counter() - rpc_start
        self.dispatch_worker_events(
            response,
            rpc_duration_s=rpc_duration_s,
        )

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        test_res = self.testing_handler(model=config["model"])
        test_res = {"executorId": self.this_rank, "results": test_res}

        # Report execution completion information.
        serialized_test_result = self.serialize_response(test_res)
        rpc_start = time.perf_counter()
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id,
                executor_id=self.executor_id,
                event=commons.MODEL_TEST,
                status=True,
                msg=None,
                meta_result=None,
                data_result=serialized_test_result,
            )
        )
        rpc_duration_s = time.perf_counter() - rpc_start
        self.dispatch_worker_events(
            response,
            rpc_duration_s=rpc_duration_s,
        )

    def Stop(self):
        """Stop the current executor"""
        logging.info(f"Terminating the executor ...")
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True
        if self.wandb != None:
            self.wandb.finish()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def override_conf(self, config):
        """Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        if conf.engine == commons.TENSORFLOW:
            from fedscale.cloud.execution.tensorflow_client import TensorflowClient
            return TensorflowClient(conf)
        elif conf.engine == commons.PYTORCH:
            if conf.task == "rl":
                return RLClient(conf)
            else:
                return TorchClient(conf)
        raise "Currently, FedScale supports tensorflow and pytorch."

    def training_handler(self, client_id, conf, model):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        self.set_received_weights(model) #, is_aggregator=False)
        conf.client_id = client_id
        conf.tokenizer = fllibs.tokenizer
        client_data = (
            self.training_sets
            if self.args.task == "rl"
            else select_dataset(
                client_id,
                self.training_sets,
                batch_size=conf.batch_size,
                args=self.args,
                collate_fn=self.collate_fn,
            )
        )
        client = self.get_client_trainer(self.args)
        gemm_trace.set_client_id(client_id)

        try:
            train_res = client.train(
                client_data=client_data,
                model=self.model_adapter.get_model(),
                conf=conf,
            )
        finally:
            gemm_trace.clear_client_id()

        return train_res

    def testing_handler(self, model):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        self.set_received_weights(model) #, is_aggregator=False)
        test_config = self.override_conf(
            {
                "rank": self.this_rank,
                "memory_capacity": self.args.memory_capacity,
                "tokenizer": fllibs.tokenizer,
            }
        )
        client = self.get_client_trainer(test_config)
        data_loader = select_dataset(
            self.this_rank,
            self.testing_sets,
            batch_size=self.args.test_bsz,
            args=self.args,
            isTest=True,
            collate_fn=self.collate_fn,
        )

        test_results = client.test(
            data_loader, model=self.model_adapter.get_model(), conf=test_config
        )
        self.log_test_result(test_results)
        gc.collect()

        return test_results

    def client_register(self):
        """Register the executor information to the aggregator"""
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                rpc_start = time.perf_counter()
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler()
                        ),
                    )
                )
                rpc_duration_s = time.perf_counter() - rpc_start
                self.dispatch_worker_events(
                    response,
                    rpc_duration_s=rpc_duration_s,
                )
                break
            except Exception as e:
                logging.warning(
                    f"Failed to connect to aggregator {e}. Will retry in 5 sec."
                )
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for a new task and record observed RPC latency."""
        rpc_start = time.perf_counter()
        response = self.aggregator_communicator.stub.CLIENT_PING(
            job_api_pb2.PingRequest(
                client_id=self.executor_id,
                executor_id=self.executor_id,
            )
        )
        rpc_duration_s = time.perf_counter() - rpc_start
        self.dispatch_worker_events(
            response,
            rpc_duration_s=rpc_duration_s,
        )

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")
        self.client_register()

        while not self.received_stop_request:
            if len(self.event_queue) > 0:
                request, response_rpc_duration_s = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    # End-to-end client update timing starts when the executor
                    # begins handling the received training task.
                    client_update_start = time.perf_counter()

                    serialized_metadata_bytes = len(request.meta)

                    train_config = self.deserialize_response(request.meta)

                    (
                        serialized_train_model,
                        download_duration_s,
                        download_effective_mbps,
                    ) = self.download_streamed_payload(
                        request.data
                    )

                    train_model = self.deserialize_response(
                        serialized_train_model
                    )

                    serialized_model_bytes = len(
                        serialized_train_model
                    )

                    client_id_for_record = int(
                        train_config["client_id"]
                    )

                    append_communication_record(
                        self.communication_log_path,
                        {
                            "round": self.round,
                            "executor_id": self.executor_id,
                            "client_id": client_id_for_record,
                            "direction": "download",
                            "payload_type": "global_model",
                            "raw_model_bytes": raw_tensor_bytes(train_model),
                            "serialized_bytes": serialized_model_bytes,
                            "metadata_bytes": serialized_metadata_bytes,
                            "method": getattr(self.args, "method", "full"),
                            "transfer_duration_s": download_duration_s,
                            "throughput_mbps": download_effective_mbps,
                            "timing_scope": (
                                "streamed_model_download"
                            ),
                        },
                    )

                    train_config["model"] = train_model
                    train_config["client_id"] = client_id_for_record

                    client_id, train_res = self.Train(train_config)

                    serialized_train_result = self.serialize_response(train_res)
                    update_weights = train_res.get("update_weight", {})

                    if getattr(self.args, "method", "full") == "topk":
                        raw_update_bytes = sum(
                            payload["indices"].nbytes
                            + payload["values"].nbytes
                            for payload in update_weights.values()
                        )
                    else:
                        raw_update_bytes = raw_tensor_bytes(update_weights)

                    upload_transfer_id = uuid.uuid4().hex
                    upload_start = time.perf_counter()

                    # Stream large client updates in bounded chunks instead of
                    # placing the entire serialized result in one gRPC message.
                    upload_chunks = iter_chunks(
                        serialized_train_result,
                        upload_transfer_id,
                        client_id=str(client_id),
                        executor_id=self.executor_id,
                        event=commons.UPLOAD_MODEL,
                    )

                    future_call = (
                        self.aggregator_communicator.stub
                        .UPLOAD_DATA
                        .future(
                            upload_chunks
                        )
                    )

                    def _upload_done_callback(
                        completed_future,
                        *,
                        upload_start=upload_start,
                        client_update_start=client_update_start,
                        client_id=int(client_id),
                        round_number=self.round,
                        serialized_upload_bytes=len(serialized_train_result),
                        raw_upload_bytes=raw_update_bytes,
                        tensor_count=len(update_weights),
                        download_duration_s=download_duration_s,
                        serialized_download_bytes=serialized_model_bytes,
                        training_duration_s=float(
                            train_res.get("real_training_duration_s", 0.0)
                        ),
                    ):
                        upload_duration_s = (
                            time.perf_counter() - upload_start
                        )
                        end_to_end_client_update_s = (
                            time.perf_counter() - client_update_start
                        )

                        upload_throughput_mbps = (
                            serialized_upload_bytes
                            * 8
                            / upload_duration_s
                            / 1_000_000
                            if upload_duration_s > 0
                            else None
                        )

                        append_communication_record(
                            self.communication_log_path,
                            {
                                "round": round_number,
                                "executor_id": self.executor_id,
                                "client_id": client_id,
                                "direction": "upload",
                                "payload_type": "client_training_result",
                                "raw_update_bytes": raw_upload_bytes,
                                "serialized_bytes": serialized_upload_bytes,
                                "tensor_count": tensor_count,
                                "method": getattr(
                                    self.args,
                                    "method",
                                    "full",
                                ),
                                "transfer_duration_s": upload_duration_s,
                                "throughput_mbps": upload_throughput_mbps,
                                "timing_scope": (
                                    "application_rpc_roundtrip_upload_completion"
                                ),
                            },
                        )

                        self._append_jsonl(
                            self.client_update_log_path,
                            {
                                "timestamp": time.time(),
                                "round": round_number,
                                "executor_id": self.executor_id,
                                "client_id": client_id,
                                "model": getattr(self.args, "model", ""),
                                "method": getattr(
                                    self.args,
                                    "method",
                                    "full",
                                ),
                                "download_duration_s": download_duration_s,
                                "real_training_duration_s": training_duration_s,
                                "upload_duration_s": upload_duration_s,
                                "end_to_end_client_update_s": (
                                    end_to_end_client_update_s
                                ),
                                "serialized_download_bytes": (
                                    serialized_download_bytes
                                ),
                                "serialized_upload_bytes": (
                                    serialized_upload_bytes
                                ),
                            },
                        )

                        try:
                            upload_reply = completed_future.result()
                        except Exception as ex:
                            logging.warning(
                                "Streamed upload RPC failed for client %s: %s",
                                client_id,
                                ex,
                            )
                            return

                        if not upload_reply.success:
                            logging.warning(
                                "Streamed upload rejected for client %s: %s",
                                client_id,
                                upload_reply.message,
                            )
                            return

                        # UPLOAD_DATA wraps the normal FedScale ServerResponse
                        # so the existing event-processing path is preserved.
                        self.dispatch_worker_events(
                            upload_reply.response,
                            rpc_duration_s=upload_duration_s,
                        )

                    future_call.add_done_callback(
                        _upload_done_callback
                    )

                elif current_event == commons.MODEL_TEST:
                    test_config = self.deserialize_response(
                        request.meta
                    )

                    (
                        serialized_test_model,
                        test_download_duration_s,
                        test_download_throughput_mbps,
                    ) = self.download_streamed_payload(
                        request.data
                    )

                    test_model = self.deserialize_response(
                        serialized_test_model
                    )

                    test_config["model"] = test_model
                    test_config["client_id"] = int(
                        test_config["client_id"]
                    )

                    append_communication_record(
                        self.communication_log_path,
                        {
                            "round": self.round,
                            "executor_id": self.executor_id,
                            "client_id": int(
                                test_config["client_id"]
                            ),
                            "direction": "download",
                            "payload_type": "test_model",
                            "raw_model_bytes": raw_tensor_bytes(
                                test_model
                            ),
                            "serialized_bytes": len(
                                serialized_test_model
                            ),
                            "method": getattr(
                                self.args,
                                "method",
                                "full",
                            ),
                            "transfer_duration_s": (
                                test_download_duration_s
                            ),
                            "throughput_mbps": (
                                test_download_throughput_mbps
                            ),
                            "timing_scope": (
                                "streamed_model_download"
                            ),
                        },
                    )

                    self.Test(test_config)

                elif current_event == commons.UPDATE_MODEL:
                    (
                        serialized_model_weights,
                        update_download_duration_s,
                        update_download_throughput_mbps,
                    ) = self.download_streamed_payload(
                        request.data
                    )

                    model_weights = self.deserialize_response(
                        serialized_model_weights
                    )

                    append_communication_record(
                        self.communication_log_path,
                        {
                            "round": self.round,
                            "executor_id": self.executor_id,
                            "client_id": None,
                            "direction": "download",
                            "payload_type": "global_model_update",
                            "raw_model_bytes": raw_tensor_bytes(
                                model_weights
                            ),
                            "serialized_bytes": len(
                                serialized_model_weights
                            ),
                            "method": getattr(
                                self.args,
                                "method",
                                "full",
                            ),
                            "transfer_duration_s": (
                                update_download_duration_s
                            ),
                            "throughput_mbps": (
                                update_download_throughput_mbps
                            ),
                            "timing_scope": (
                                "streamed_model_download"
                            ),
                        },
                    )

                    self.UpdateModel(model_weights)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                try:
                    self.client_ping()
                except Exception as e:
                    logging.info(
                        f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ..."
                    )
                    self.Stop()

    def log_test_result(self, test_res):
        """Log test results to wandb server if enabled"""
        acc = round(test_res["top_1"] / test_res["test_len"], 4)
        acc_5 = round(test_res["top_5"] / test_res["test_len"], 4)
        test_loss = test_res["test_loss"] / test_res["test_len"]
        if self.wandb != None:
            self.wandb.log(
                {
                    "Test/round_to_top1_accuracy": acc,
                    "Test/round_to_top5_accuracy": acc_5,
                    "Test/round_to_loss": test_loss,
                },
                step=self.round,
            )


if __name__ == "__main__":
    executor = Executor(parser.args)
    executor.run()
