from typing import List

import numpy as np
import torch
import copy
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.model_adapter_base import ModelAdapterBase


class TorchModelAdapter(ModelAdapterBase):
    """
    Adapts functions to pytorch models.
    """
    def __init__(self, model: torch.nn.Module, optimizer: TorchServerOptimizer = None):
        """
        Initializes a TorchModelAdapter.
        :param model: the PyTorch model to adapt
        :param optimizer: the optimizer to apply weights, when specified.
        """
        self.model = model
        self.optimizer = optimizer

    def set_weights(
        self,
        weights,
        is_aggregator=True,
        client_training_results=None,
    ):
        """
        Set model weights.

        Supports:
        - Standard FedScale full-model weights as an ordered list.
        - LoRA/PEFT adapter weights as a name-keyed dictionary.
        """

        # LoRA weights are transmitted as a dictionary keyed by parameter name.
        if isinstance(weights, dict):
            self.set_lora_weights(weights)
            return

        # Standard FedScale full-model behaviour.
        last_grad_weights = [
            param.data.clone()
            for param in self.model.state_dict().values()
        ]

        if len(weights) != len(self.model.state_dict()):
            raise ValueError(
                f"Received {len(weights)} weights, but model state_dict "
                f"contains {len(self.model.state_dict())} entries."
            )

        new_state_dict = {
            name: torch.from_numpy(
                np.asarray(weights[i], dtype=np.float32)
            ).to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for i, (name, tensor) in enumerate(
                self.model.state_dict().items()
            )
        }

        self.model.load_state_dict(new_state_dict)

        if self.optimizer and is_aggregator:
            weights_origin = copy.deepcopy(weights)

            tensor_weights = [
                torch.tensor(x)
                for x in weights_origin
            ]

            self.optimizer.update_round_gradient(
                last_grad_weights,
                tensor_weights,
                self.model,
                client_training_results,
            )

    def set_lora_weights(self, weights):
        """Apply a dictionary of LoRA adapter weights to a PEFT model."""
        from peft import set_peft_model_state_dict

        state_dict = {
            name: torch.from_numpy(
                np.asarray(value, dtype=np.float32)
            )
            for name, value in weights.items()
        }

        set_peft_model_state_dict(
            self.model,
            state_dict,
        )

    def get_lora_weights(self):
        """Return only LoRA adapter weights as a named dictionary."""
        from peft import get_peft_model_state_dict

        state_dict = get_peft_model_state_dict(self.model)

        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in state_dict.items()
        }

    def get_weights(self) -> List[np.ndarray]:
        """
        Get the model's weights as a numpy weights array. Note that it doesn't contain layer names. Rather, index 0
        contains the model's first layer weights, and index N contains the N+1 layer's weights.
        :return: A numpy array
        """
        return [params.data.clone() for params in self.model.state_dict().values()]

    def apply_delta(self, delta_weights):
        """Apply an averaged model delta to the current global model."""

        current_state = self.model.state_dict()

        new_state = {}

        for name, tensor in current_state.items():

            if name in delta_weights:

                delta = torch.from_numpy(
                    np.asarray(
                        delta_weights[name],
                        dtype=np.float32,
                    )
                ).to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )

                new_state[name] = (
                    tensor.detach()
                    + delta
                )

            else:

                new_state[name] = tensor

        self.model.load_state_dict(
            new_state,
            strict=True,
        )

    def get_model(self):
        """
        Get the instantiated framework specific model including the architecture.
        """
        return self.model
