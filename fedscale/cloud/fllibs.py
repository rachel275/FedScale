# Standard libs
import json
import logging
import os
import sys
import torchvision.models as tormodels
from torchvision import datasets, transforms

# libs from fedscale
import fedscale.cloud.config_parser as parser
from fedscale.cloud import commons
from fedscale.dataloaders.utils_data import get_data_transform
# FedScale model libs
from fedscale.utils.models.torch_model_provider import get_cv_model
#from fedscale.utils.models.tensorflow_model_provider import get_tensorflow_model

tokenizer = None


def import_libs():
    global tokenizer

    if parser.args.task == 'nlp' or parser.args.task == 'text_clf':
        global AdamW, AutoConfig, AutoTokenizer
        global AutoModelForMaskedLM, AutoModelForCausalLM
        global load_and_cache_examples, mask_tokens

        from torch.optim import AdamW
        

        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
        )

        from fedscale.dataloaders.nlp import (
            load_and_cache_examples,
            mask_tokens,
        )
    elif parser.args.task == 'speech':
        global numba, SPEECH, BackgroundNoiseDataset, AddBackgroundNoiseOnSTFT, DeleteSTFT, FixSTFTDimension, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, ToMelSpectrogramFromSTFT, ToSTFT, ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor

        import numba

        from fedscale.dataloaders.speech import SPEECH, BackgroundNoiseDataset
        from fedscale.dataloaders.transforms_stft import (AddBackgroundNoiseOnSTFT,
                                                          DeleteSTFT,
                                                          FixSTFTDimension,
                                                          StretchAudioOnSTFT,
                                                          TimeshiftAudioOnSTFT,
                                                          ToMelSpectrogramFromSTFT,
                                                          ToSTFT)
        from fedscale.dataloaders.transforms_wav import (ChangeAmplitude,
                                                         ChangeSpeedAndPitchAudio,
                                                         FixAudioLength, LoadAudio,
                                                         ToMelSpectrogram,
                                                         ToTensor)
    elif parser.args.task == 'detection':
        global pickle, get_imdb, readClass, resnet, nms, bbox_transform_inv, clip_boxes, cfg, cfg_from_file, cfg_from_list, get_output_dir, adjust_learning_rate, clip_gradient, load_net, save_checkpoint, save_net, weights_normal_init, roibatchLoader, combined_roidb

        import pickle
        from fedscale.dataloaders.rcnn.lib.datasets.factory import get_imdb
        from fedscale.dataloaders.rcnn.lib.datasets.pascal_voc import readClass
        from fedscale.dataloaders.rcnn.lib.model.faster_rcnn.resnet import resnet
        from fedscale.dataloaders.rcnn.lib.model.roi_layers import nms
        from fedscale.dataloaders.rcnn.lib.model.rpn.bbox_transform import (
            bbox_transform_inv, clip_boxes)
        from fedscale.dataloaders.rcnn.lib.model.utils.config import (
            cfg, cfg_from_file, cfg_from_list, get_output_dir)
        from fedscale.dataloaders.rcnn.lib.model.utils.net_utils import (
            adjust_learning_rate, clip_gradient, load_net, save_checkpoint,
            save_net, weights_normal_init)
        from fedscale.dataloaders.rcnn.lib.roi_data_layer.roibatchLoader import \
            roibatchLoader
        from fedscale.dataloaders.rcnn.lib.roi_data_layer.roidb import \
            combined_roidb
    elif parser.args.task == 'voice':
        global CTCLoss

        from torch_baidu_ctc import CTCLoss
    elif parser.args.task == 'rl':
        global gym, RLData, Net, DQN

        import gym

        from fedscale.dataloaders.dqn import RLData, Net, DQN


# shared functions of aggregator and clients
# initiate for nlp

# Yile: are these vars used anywhere?
os.environ['MASTER_ADDR'] = parser.args.ps_ip
os.environ['MASTER_PORT'] = parser.args.ps_port

outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47, 'amazon': 5,
               'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist': 1010
               }

def migrate_model_to_dgemm_arena(model):
    """
    Move all CPU parameter and buffer storage into the DGEMM shared arena.

    PyTorch tensors remain ordinary device='cpu' tensors. Parameters are
    migrated one at a time so that old checkpoint-backed storage can be
    released progressively rather than duplicating the entire model at once.
    """
    import gc
    import torch
    import torch_dcpu

    total_bytes = 0
    migrated_bytes = 0
    already_arena_bytes = 0
    migrated_params = 0
    migrated_buffers = 0

    # Parameters. recurse=False is important because we walk every module
    # ourselves and must not process a parameter multiple times.
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param is None:
                continue

            nbytes = param.numel() * param.element_size()
            total_bytes += nbytes

            if torch.ops.torch_dcpu.ptr_in_arena(param):
                already_arena_bytes += nbytes
                continue

            old_data = param.data

            new_data = torch.ops.torch_dcpu.arena_clone_cpu(
                old_data.detach().contiguous()
            )

            # Preserve the Parameter object itself. This is preferable to
            # replacing it because other model structures may hold references
            # to the Parameter.
            param.data = new_data

            migrated_bytes += nbytes
            migrated_params += 1

            del old_data
            del new_data

        # Buffers are not Parameters, so replace the module's registered
        # buffer tensor directly.
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is None:
                continue

            nbytes = buf.numel() * buf.element_size()
            total_bytes += nbytes

            if torch.ops.torch_dcpu.ptr_in_arena(buf):
                already_arena_bytes += nbytes
                continue

            new_buf = torch.ops.torch_dcpu.arena_clone_cpu(
                buf.detach().contiguous()
            )

            module._buffers[name] = new_buf

            migrated_bytes += nbytes
            migrated_buffers += 1

    gc.collect()

    logging.info(
        "DGEMM arena migration complete: "
        "%d parameters, %d buffers migrated; "
        "%.3f GiB migrated, %.3f GiB already arena-backed, "
        "%.3f GiB total tensor storage",
        migrated_params,
        migrated_buffers,
        migrated_bytes / (1024 ** 3),
        already_arena_bytes / (1024 ** 3),
        total_bytes / (1024 ** 3),
    )

    return model

def init_model():
    global tokenizer

    logging.info("Initializing the model ...")

    import_libs()

    if parser.args.task == "nlp":
        from transformers import (
            AutoModelForMaskedLM,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        model_name = parser.args.model
        model_name_lower = model_name.lower()
        hf_token = os.environ.get("HF_TOKEN")
        
        # Decoder-only / causal language models.
        is_causal_lm = any(
            name in model_name_lower
            for name in (
                "llama",
                "qwen",
                "mistral",
                "gemma",
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            token=hf_token,
        )

        # Install the shared-arena allocator only after tokenizer construction,
        # but before model construction. From this point onward ordinary PyTorch
        # CPU tensor storage is allocated directly from the DGEMM arena.
        if getattr(parser.args, "use_dcpu", False):
            import torch
            import torch_dcpu

            torch.ops.torch_dcpu.install_arena_cpu_allocator()

            logging.info(
                "Installed DGEMM shared arena as PyTorch CPU tensor allocator"
            )

        if is_causal_lm:
            if parser.args.method == "qlora":
                import torch
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float32,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    quantization_config=quantization_config,
                )
            else:
                import torch

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
            
            # Hugging Face's low-memory checkpoint loader can construct parameter
            # storage without going through PyTorch's normal CPU allocator. Migrate
            # that pretrained storage into the shared DGEMM arena once loading has
            # completed.
            if getattr(parser.args, "use_dcpu", False):
                logging.info(
                    "Migrating pretrained model storage into DGEMM shared arena"
                )

                model = migrate_model_to_dgemm_arena(model)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.config.pad_token_id = tokenizer.pad_token_id

        else:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                token=hf_token,
            )


        # Apply LoRA / QLoRA AFTER loading either model type
        if parser.args.method in ("lora", "qlora"):
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
            )

            if parser.args.method == "qlora":
                from peft import prepare_model_for_kbit_training

                logging.info(
                    "Preparing quantized model for QLoRA training"
                )

                model = prepare_model_for_kbit_training(model)

            if "distilbert" in model_name_lower:
                target_modules = [
                    "q_lin",
                    "v_lin",
                ]

            elif is_causal_lm:
                target_modules = [
                    "q_proj",
                    "v_proj",
                ]

            else:
                target_modules = [
                    "query",
                    "value",
                ]

            if is_causal_lm:
                peft_task_type = TaskType.CAUSAL_LM
            else:
                peft_task_type = TaskType.FEATURE_EXTRACTION

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules,
                task_type=peft_task_type,
            )

            model = get_peft_model(
                model,
                lora_config,
            )

            # Verify that model parameters are backed by the DGEMM arena.
            if getattr(parser.args, "use_dcpu", False):
                total_params = 0
                total_bytes = 0
                arena_params = 0
                arena_bytes = 0

                for p in model.parameters():
                    nbytes = p.numel() * p.element_size()

                    total_params += p.numel()
                    total_bytes += nbytes

                    if torch.ops.torch_dcpu.ptr_in_arena(p):
                        arena_params += p.numel()
                        arena_bytes += nbytes

                logging.info(
                    "DGEMM arena parameter coverage: "
                    "%d/%d params (%.2f%%), "
                    "%.3f/%.3f GiB (%.2f%% bytes)",
                    arena_params,
                    total_params,
                    100.0 * arena_params / max(total_params, 1),
                    arena_bytes / (1024 ** 3),
                    total_bytes / (1024 ** 3),
                    100.0 * arena_bytes / max(total_bytes, 1),
                )

            if getattr(parser.args, "use_dcpu", False):
                from fedscale.cloud.execution.dcpu_linear import (
                    route_peft_base_linears,
                )

                routed = route_peft_base_linears(model)

                logging.info(
                    "DCPU CPU-resident routing installed for %d PEFT base Linear layers",
                    routed,
                )

            model.print_trainable_parameters()

    elif parser.args.task == 'text_clf':

        if parser.args.model == 'albert':
            from transformers import AlbertForSequenceClassification
            config = AutoConfig.from_pretrained(os.path.join(
                parser.args.log_path, 'albert-small-config.json'))
            config.num_labels = outputClass[parser.args.data_set]
            model = AlbertForSequenceClassification(config)
        elif parser.args.model == 'lr':
            from fedscale.utils.models.simple.models import LogisticRegression
            model = LogisticRegression(300, outputClass[parser.args.data_set])

    elif parser.args.task == 'tag-one-sample':
        # Load LR model for tag prediction
        model = LogisticRegression(parser.args.vocab_token_size, parser.args.vocab_tag_size)
    elif parser.args.task == 'speech':
        if parser.args.model == 'mobilenet':
            from fedscale.utils.models.specialized.resnet_speech import \
                mobilenet_v2
            model = mobilenet_v2(num_classes=outputClass[parser.args.data_set])
        elif parser.args.model == "resnet18":
            from fedscale.utils.models.specialized.resnet_speech import \
                resnet18
            model = resnet18(
                num_classes=outputClass[parser.args.data_set], in_channels=1)
        elif parser.args.model == "resnet34":
            from fedscale.utils.models.specialized.resnet_speech import \
                resnet34
            model = resnet34(
                num_classes=outputClass[parser.args.data_set], in_channels=1)
        elif parser.args.model == "resnet50":
            from fedscale.utils.models.specialized.resnet_speech import \
                resnet50
            model = resnet50(
                num_classes=outputClass[parser.args.data_set], in_channels=1)
        elif parser.args.model == "resnet101":
            from fedscale.utils.models.specialized.resnet_speech import \
                resnet101
            model = resnet101(
                num_classes=outputClass[parser.args.data_set], in_channels=1)
        elif parser.args.model == "resnet152":
            from fedscale.utils.models.specialized.resnet_speech import \
                resnet152
            model = resnet152(
                num_classes=outputClass[parser.args.data_set], in_channels=1)
        else:
            # Should not reach here
            logging.info('Model must be resnet or mobilenet')
            sys.exit(-1)

    elif parser.args.task == 'voice':
        from fedscale.utils.models.specialized.voice_model import (
            DeepSpeech, supported_rnns)

        # Initialise new model training
        with open(os.path.join(parser.args.data_dir, "labels.json")) as label_file:
            labels = json.load(label_file)

        audio_conf = dict(sample_rate=parser.args.sample_rate,
                          window_size=parser.args.window_size,
                          window_stride=parser.args.window_stride,
                          window=parser.args.window,
                          noise_dir=parser.args.noise_dir,
                          noise_prob=parser.args.noise_prob,
                          noise_levels=(parser.args.noise_min, parser.args.noise_max))
        model = DeepSpeech(rnn_hidden_size=parser.args.hidden_size,
                           nb_layers=parser.args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[parser.args.rnn_type.lower()],
                           audio_conf=audio_conf,
                           bidirectional=parser.args.bidirectional)
    elif parser.args.task == 'detection':
        cfg_from_file(parser.args.cfg_file)
        cfg_from_list(['DATA_DIR', parser.args.data_dir])
        model = resnet(readClass(os.path.join(parser.args.data_dir, "class.txt")), 50,
                       pretrained=True, class_agnostic=False, pretrained_model=parser.args.backbone)
        model.create_architecture()
        return model
    elif parser.args.task == 'rl':
        model = DQN(parser.args).target_net
    else:
        if parser.args.model == "lr":
            from fedscale.utils.models.simple.models import LogisticRegression
            model = LogisticRegression(
                parser.args.input_dim, outputClass[parser.args.data_set])
        elif parser.args.model == 'svm':
            from fedscale.utils.models.simple.models import LinearSVM
            model = LinearSVM(parser.args.input_dim, outputClass[parser.args.data_set])
        elif parser.args.model_zoo == "fedscale-tensorflow-zoo":
            assert parser.args.engine == commons.TENSORFLOW
            from fedscale.utils.models.tensorflow_model_provider import (
                get_tensorflow_model,
            )

            model = get_tensorflow_model(parser.args.model, parser.args)
        else:
            if parser.args.model_zoo == "fedscale-torch-zoo":
                if parser.args.task == "cv":
                    model = get_cv_model(name=parser.args.model,
                                         num_classes=outputClass[parser.args.data_set])
                else:
                    raise NameError(f"Model zoo {parser.args.model_zoo} does not exist")
            elif parser.args.model_zoo == "torchcv":
                model = tormodels.__dict__[parser.args.model](
                    num_classes=outputClass[parser.args.data_set])
            else:
                raise NameError(f"Model zoo {parser.args.model_zoo} does not exist")
    return model


def init_dataset():
    import_libs()

    if parser.args.task == "detection":
        if not os.path.exists(parser.args.data_cache):
            imdb_name = "voc_2007_trainval"
            imdbval_name = "voc_2007_test"
            imdb, roidb, ratio_list, ratio_index = combined_roidb(
                imdb_name, ['DATA_DIR', parser.args.data_dir], sizes=parser.args.train_size_file)
            train_dataset = roibatchLoader(
                roidb, ratio_list, ratio_index, parser.args.batch_size, imdb.num_classes, imdb._image_index_temp,
                training=True)
            imdb_, roidb_, ratio_list_, ratio_index_ = combined_roidb(
                imdbval_name, ['DATA_DIR', parser.args.data_dir], sizes=parser.args.test_size_file, training=False)
            imdb_.competition_mode(on=True)
            test_dataset = roibatchLoader(roidb_, ratio_list_, ratio_index_, 1,
                                          imdb_.num_classes, imdb_._image_index_temp, training=False, normalize=False)
            with open(parser.args.data_cache, 'wb') as f:
                pickle.dump(train_dataset, f, -1)
                pickle.dump(test_dataset, f, -1)
        else:
            with open(parser.args.data_cache, 'rb') as f:
                train_dataset = pickle.load(f)
                test_dataset = pickle.load(f)
    elif parser.args.task == "rl":
        train_dataset = test_dataset = RLData(parser.args)
    else:

        if parser.args.data_set == 'Mnist':
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(parser.args.data_dir, train=True, download=True,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(parser.args.data_dir, train=False, download=True,
                                          transform=test_transform)

        elif parser.args.data_set == 'cifar10':
            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(parser.args.data_dir, train=True, download=True,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(parser.args.data_dir, train=False, download=True,
                                            transform=test_transform)

        elif parser.args.data_set == "imagenet":
            train_transform, test_transform = get_data_transform('imagenet')
            train_dataset = datasets.ImageNet(
                parser.args.data_dir, split='train', download=False, transform=train_transform)
            test_dataset = datasets.ImageNet(
                parser.args.data_dir, split='val', download=False, transform=test_transform)

        elif parser.args.data_set == 'emnist':
            test_dataset = datasets.EMNIST(
                parser.args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
            train_dataset = datasets.EMNIST(
                parser.args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

        elif parser.args.data_set == 'femnist':
            from fedscale.dataloaders.femnist import FEMNIST

            train_transform, test_transform = get_data_transform('mnist')
            train_dataset = FEMNIST(
                parser.args.data_dir, dataset='train', transform=train_transform)
            test_dataset = FEMNIST(
                parser.args.data_dir, dataset='test', transform=test_transform)

        elif parser.args.data_set == 'openImg':
            from fedscale.dataloaders.openimage import OpenImage

            train_transform, test_transform = get_data_transform('openImg')
            train_dataset = OpenImage(
                parser.args.data_dir, dataset='train', transform=train_transform)
            test_dataset = OpenImage(
                parser.args.data_dir, dataset='test', transform=test_transform)

        elif parser.args.data_set == 'blog':
            train_dataset = load_and_cache_examples(
                parser.args, tokenizer, evaluate=False)
            test_dataset = load_and_cache_examples(
                parser.args, tokenizer, evaluate=True)

        elif parser.args.data_set == 'stackoverflow':
            from fedscale.dataloaders.stackoverflow import stackoverflow

            train_dataset = stackoverflow(parser.args.data_dir, train=True)
            test_dataset = stackoverflow(parser.args.data_dir, train=False)

        elif parser.args.data_set == 'amazon':
            if parser.args.model == 'albert':
                import fedscale.dataloaders.amazon as fl_loader
                train_dataset = fl_loader.AmazonReview_loader(
                    parser.args.data_dir, train=True, tokenizer=tokenizer, max_len=parser.args.clf_block_size)
                test_dataset = fl_loader.AmazonReview_loader(
                    parser.args.data_dir, train=False, tokenizer=tokenizer, max_len=parser.args.clf_block_size)

            elif parser.args.model == 'lr':
                import fedscale.dataloaders.word2vec as fl_loader
                train_dataset = fl_loader.AmazonReview_word2vec(
                    parser.args.data_dir, parser.args.embedding_file, train=True)
                test_dataset = fl_loader.AmazonReview_word2vec(
                    parser.args.data_dir, parser.args.embedding_file, train=False)

        elif parser.args.data_set == 'yelp':
            import fedscale.dataloaders.yelp as fl_loader

            train_dataset = fl_loader.TextSentimentDataset(
                parser.args.data_dir, train=True, tokenizer=tokenizer, max_len=parser.args.clf_block_size)
            test_dataset = fl_loader.TextSentimentDataset(
                parser.args.data_dir, train=False, tokenizer=tokenizer, max_len=parser.args.clf_block_size)

        elif parser.args.data_set == 'google_speech':
            bkg = '_background_noise_'
            data_aug_transform = transforms.Compose(
                [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
                 TimeshiftAudioOnSTFT(), FixSTFTDimension()])
            bg_dataset = BackgroundNoiseDataset(
                os.path.join(parser.args.data_dir, bkg), data_aug_transform)
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
                n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
            train_dataset = SPEECH(parser.args.data_dir, dataset='train',
                                   transform=transforms.Compose([LoadAudio(),
                                                                 data_aug_transform,
                                                                 add_bg_noise,
                                                                 train_feature_transform]))
            valid_feature_transform = transforms.Compose(
                [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
            test_dataset = SPEECH(parser.args.data_dir, dataset='test',
                                  transform=transforms.Compose([LoadAudio(),
                                                                FixAudioLength(),
                                                                valid_feature_transform]))
        elif parser.args.data_set == 'common_voice':
            from fedscale.dataloaders.voice_data_loader import \
                SpectrogramDataset
            train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                               data_dir=parser.args.data_dir,
                                               labels=model.labels,
                                               train=True,
                                               normalize=True,
                                               speed_volume_perturb=parser.args.speed_volume_perturb,
                                               spec_augment=parser.args.spec_augment,
                                               data_mapfile=parser.args.data_mapfile)
            test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                              data_dir=parser.args.data_dir,
                                              labels=model.labels,
                                              train=False,
                                              normalize=True,
                                              speed_volume_perturb=False,
                                              spec_augment=False)
        else:
            logging.info('DataSet must be {}!'.format(
                ['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
            sys.exit(-1)

    return train_dataset, test_dataset
