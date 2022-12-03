'''Simple script to finetune a stable-diffusion model'''

import argparse
import contextlib
import copy
import gc
import hashlib
import itertools
import json
import math
import os
import re
import random
import shutil
import subprocess
import time
import atexit
import zipfile
import tempfile
import multiprocessing
from pathlib import Path
from contextlib import nullcontext
from urllib.parse import urlparse
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torch.hub import download_url_to_file, get_dir

try:
    # pip install git+https://github.com/KichangKim/DeepDanbooru
    import tensorflow as tf
    import deepdanbooru as dd

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except ImportError:
    pass

try:
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
except Exception:
    pass

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import (
    get_scheduler,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="",
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--class_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images,"
            "additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation "
            "dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--scale_lr_sqrt",
        action="store_true",
        default=False,
        help="Scale the learning rate using sqrt instead of linear method.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "cosine_with_restarts_mod", "cosine_mod"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_deepspeed_adam", action="store_true", help="Whether or not to use deepspeed Adam."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adamw_8bit", "adamw_ds", "sgdm", "sgdm_8bit"],
        help=(
            "The optimizer to use. _8bit optimizers require bitsandbytes, _ds optimizers require deepspeed."
        )
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="Momentum value for the SGDM optimizer"
    )
    parser.add_argument(
        "--sgd_dampening",
        type=float,
        default=0,
        help="Dampening value for the SGDM optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10_000,
        help="Save weights every N steps."
    )
    parser.add_argument(
        "--save_min_steps",
        type=int,
        default=10,
        help="Start saving weights after N steps."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--not_cache_latents",
        action="store_true",
        help="Do not precompute and cache latents from VAE."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Use wandb to watch training process.",
    )
    parser.add_argument(
        "--wandb_artifact",
        default=False,
        action="store_true",
        help="Upload saved weights to wandb.",
    )
    parser.add_argument(
        "--rm_after_wandb_saved",
        default=False,
        action="store_true",
        help="Remove saved weights from local machine after uploaded to wandb. Useful in colab.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="Stable-Diffusion-Dreambooth",
        help="Project name in your wandb.",
    )
    parser.add_argument(
        "--read_prompt_filename",
        default=False,
        action="store_true",
        help="Append extra prompt from filename.",
    )
    parser.add_argument(
        "--read_prompt_txt",
        default=False,
        action="store_true",
        help="Append extra prompt from txt.",
    )
    parser.add_argument(
        "--append_prompt",
        type=str,
        default="instance",
        choices=["class", "instance", "both"],
        help="Append extra prompt to which part of input.",
    )
    parser.add_argument(
        "--save_unet_half",
        default=False,
        action="store_true",
        help="Use half precision to save unet weights, saves storage.",
    )
    parser.add_argument(
        "--unet_half",
        default=False,
        action="store_true",
        help="Use half precision to save unet weights, saves storage.",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=1,
        help="Stop At last [n] layers of CLIP model when training."
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=1,
        help="The number of hard restarts to use. Only works with --lr_scheduler=[cosine_with_restarts_mod, cosine_mod]"
    )
    parser.add_argument(
        "--last_epoch",
        type=int,
        default=-1,
        help="The index of the last epoch when resuming training. Only works with --lr_scheduler=[cosine_with_restarts_mod, cosine_mod]"
    )
    parser.add_argument(
        "--use_aspect_ratio_bucket",
        default=False,
        action="store_true",
        help="Use aspect ratio bucketing as image processing strategy, which may improve the quality of outputs. Use it with --not_cache_latents"
    )
    parser.add_argument(
        "--debug_arb",
        default=False,
        action="store_true",
        help="Enable debug logging on aspect ratio bucket."
    )
    parser.add_argument(
        "--save_optimizer",
        default=True,
        action="store_true",
        help="Save optimizer and scheduler state dict when training. Deprecated: use --save_states"
    )
    parser.add_argument(
        "--save_states",
        default=True,
        action="store_true",
        help="Save optimizer and scheduler state dict when training."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Load optimizer and scheduler state dict to continue training."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Specify checkpoint to resume. Use wandb://[artifact-full-name] for wandb artifact."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Read args from config file. Command line args have higher priority and will override it.",
    )
    parser.add_argument(
        "--arb_dim_limit",
        type=int,
        default=1024,
        help="Aspect ratio bucketing arguments: dim_limit."
    )
    parser.add_argument(
        "--arb_divisible",
        type=int,
        default=64,
        help="Aspect ratio bucketing arguments: divisbile."
    )
    parser.add_argument(
        "--arb_max_ar_error",
        type=int,
        default=4,
        help="Aspect ratio bucketing arguments: max_ar_error."
    )
    parser.add_argument(
        "--arb_max_size",
        type=int,
        nargs="+",
        default=(768, 512),
        help="Aspect ratio bucketing arguments: max_size. example: --arb_max_size 768 512"
    )
    parser.add_argument(
        "--arb_min_dim",
        type=int,
        default=256,
        help="Aspect ratio bucketing arguments: min_dim."
    )
    parser.add_argument(
        "--deepdanbooru",
        default=False,
        action="store_true",
        help="Use deepdanbooru to tag images when prompt txt is not available."
    )
    parser.add_argument(
        "--dd_threshold",
        type=float,
        default=0.6,
        help="Threshold for Deepdanbooru tag estimation"
    )
    parser.add_argument(
        "--dd_alpha_sort",
        default=False,
        action="store_true",
        help="Sort deepbooru tags alphabetically."
    )
    parser.add_argument(
        "--dd_use_spaces",
        default=True,
        action="store_true",
        help="Use spaces for tags in deepbooru."
    )
    parser.add_argument(
        "--dd_use_escape",
        default=True,
        action="store_true",
        help="Use escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"
    )
    parser.add_argument(
        "--enable_rotate",
        default=False,
        action="store_true",
        help="Enable experimental feature to rotate image when buckets is not fit."
    )
    parser.add_argument(
        "--dd_include_ranks",
        default=False,
        action="store_true",
        help="Include rank tag in deepdanbooru."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--ucg",
        type=float,
        default=0.0,
        help="Percentage chance of dropping out the text condition per batch. \
            Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout."
    )
    parser.add_argument(
        "--debug_prompt",
        default=False,
        action="store_true",
        help="Print input prompt when training."
    )
    parser.add_argument(
        "--xformers",
        default=False,
        action="store_true",
        help="Enable memory efficient attention when training."
    )
    parser.add_argument(
        "--reinit_scheduler",
        default=False,
        action="store_true",
        help="Reinit scheduler when resume training."
    )

    args = parser.parse_args()
    resume_from = args.resume_from

    if resume_from.startswith("wandb://"):
        import wandb
        run = wandb.init(project=args.wandb_name, reinit=False)
        artifact = run.use_artifact(resume_from.replace("wandb://", ""), type='model')
        resume_from = artifact.download()

    elif args.resume_from != "":
        fp = os.path.join(resume_from, "state.pt")
        if not Path(fp).is_file():
            raise ValueError(f"State_dict file {fp} not found.")

    elif args.resume:
        rx = re.compile(r'checkpoint_(\d+)')
        ckpts = rx.findall(" ".join([x.name for x in Path(args.output_dir).iterdir() if x.is_dir() and rx.match(x.name)]))

        if not any(ckpts):
            raise ValueError("At least one model is needed to resume training.")

        ckpts.sort(key=lambda e: int(e), reverse=True)
        for k in ckpts:
            fp = os.path.join(args.output_dir, f"checkpoint_{k}", "state.pt")
            if Path(fp).is_file():
                resume_from = os.path.join(args.output_dir, f"checkpoint_{k}")
                break


        print(f"[*] Selected {resume_from}. To specify other checkpoint, use --resume-from")

    if resume_from:
        args.config = os.path.join(resume_from, "args.json")

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        parser.set_defaults(**config)
        args = parser.parse_args()

    if args.resume:
        args.pretrained_model_name_or_path = resume_from

    if not args.pretrained_model_name_or_path or not Path(args.pretrained_model_name_or_path).is_dir():
        raise ValueError("A model is needed.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DeepDanbooru:
    def __init__(
        self,
        dd_threshold=0.6,
        dd_alpha_sort=False,
        dd_use_spaces=True,
        dd_use_escape=True,
        dd_include_ranks=False,
        **kwargs
    ):

        self.threshold = dd_threshold
        self.alpha_sort = dd_alpha_sort
        self.use_spaces = dd_use_spaces
        self.use_escape = dd_use_escape
        self.include_ranks = dd_include_ranks
        self.re_special = re.compile(r"([\\()])")
        self.new_process()

    def get_tags_local(self,image):
        self.returns["value"] = -1
        self.queue.put(image)
        while self.returns["value"] == -1:
            time.sleep(0.1)

        return self.returns["value"]

    def deepbooru_process(self):
        import tensorflow, deepdanbooru
        print(f"Deepdanbooru initialized using threshold: {self.threshold}")
        self.load_model()
        while True:
            image = self.queue.get()
            if image == "QUIT":
                break
            else:
                self.returns["value"] = self.get_tags(image)

    def new_process(self):
        context = multiprocessing.get_context("spawn")
        manager = context.Manager()
        self.queue = manager.Queue()
        self.returns = manager.dict()
        self.returns["value"] = -1
        self.process = context.Process(target=self.deepbooru_process)
        self.process.start()

    def kill_process(self):
        self.queue.put("QUIT")
        self.process.join()
        self.queue = None
        self.returns = None
        self.process = None

    def load_model(self):
        model_path = Path(tempfile.gettempdir()) / "deepbooru"
        if not Path(model_path / "project.json").is_file():
            self.load_file_from_url(r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip", model_path)

            with zipfile.ZipFile(model_path / "deepdanbooru-v3-20211112-sgd-e28.zip", "r") as zip_ref:
                zip_ref.extractall(model_path)
            os.remove(model_path / "deepdanbooru-v3-20211112-sgd-e28.zip")

        self.tags = dd.project.load_tags_from_project(model_path)
        self.model = dd.project.load_model_from_project(model_path, compile_model=False)

    def unload_model(self):
        self.kill_process()

        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        tf.keras.backend.clear_session()

    @staticmethod
    def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
        if model_dir is None:  # use the pytorch hub_dir
            hub_dir = get_dir()
            model_dir = os.path.join(hub_dir, 'checkpoints')

        os.makedirs(model_dir, exist_ok=True)

        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_name is not None:
            filename = file_name
        cached_file = os.path.abspath(os.path.join(model_dir, filename))
        if not os.path.exists(cached_file):
            print(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
        return cached_file

    def process_img(self, image):
        width = self.model.input_shape[2]
        height = self.model.input_shape[1]
        image = np.array(image)
        image = tf.image.resize(
            image,
            size=(height, width),
            method=tf.image.ResizeMethod.BICUBIC,
            preserve_aspect_ratio=True,
        )
        image = image.numpy()  # EagerTensor to np.array
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0
        image_shape = image.shape
        image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
        return image

    def process_tag(self, y):
        result_dict = {}

        for i, tag in enumerate(self.tags):
            result_dict[tag] = y[i]

        unsorted_tags_in_theshold = []
        result_tags_print = []
        for tag in self.tags:
            if result_dict[tag] >= self.threshold:
                if tag.startswith("rating:"):
                    continue
                unsorted_tags_in_theshold.append((result_dict[tag], tag))
                result_tags_print.append(f"{result_dict[tag]} {tag}")

        # sort tags
        result_tags_out = []
        sort_ndx = 0
        if self.alpha_sort:
            sort_ndx = 1

        # sort by reverse by likelihood and normal for alpha, and format tag text as requested
        unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=(not self.alpha_sort))
        for weight, tag in unsorted_tags_in_theshold:
            tag_outformat = tag
            if self.use_spaces:
                tag_outformat = tag_outformat.replace("_", " ")
            if self.use_escape:
                tag_outformat = re.sub(self.re_special, r"\\\1", tag_outformat)
            if self.include_ranks:
                tag_outformat = f"({tag_outformat}:{weight:.3f})"

            result_tags_out.append(tag_outformat)

        # print("\n".join(sorted(result_tags_print, reverse=True)))

        return ", ".join(result_tags_out)

    def get_tags(self, image):
        result = self.model.predict(self.process_img(image))[0]
        return self.process_tag(result)


class AspectRatioBucket:
    '''
    Code from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py

    BucketManager impls NovelAI Aspect Ratio Bucketing, which may greatly improve the quality of outputs according to Novelai's blog (https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac)
    Requires a pickle with mapping of dataset IDs to resolutions called resolutions.pkl to use this.
    '''

    def __init__(self,
        id_size_map,
        max_size=(768, 512),
        divisible=64,
        step_size=8,
        min_dim=256,
        base_res=(512, 512),
        bsz=1,
        world_size=1,
        global_rank=0,
        max_ar_error=4,
        seed=42,
        dim_limit=1024,
        debug=True,
    ):
        if global_rank == -1:
            global_rank = 0
            
        self.res_map = id_size_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = self.get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)

        # separate prng for sharding use for increased thread resilience
        self.epoch_prng = self.get_prng(epoch_seed)
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    @staticmethod
    def get_prng(seed):
        return np.random.RandomState(seed)
    
    def __len__(self):
        return len(self.res_map) // self.bsz

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(
            res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(
            list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            try:
                print(f"skipped images: {skipped}")
                print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
                for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                    print(
                        f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
                print(f"assign_buckets: {timer:.5f}s")
            except Exception as e:
                pass

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = sorted(list(self.res_map.keys()))
        index_len = len(index)
        
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        # if self.debug:
            # print("perm", self.global_rank, index[0:16])
        
        index = index[self.global_rank::self.world_size]
        self.batch_total = len(index) // self.bsz
        assert (len(index) % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = [post_id for post_id in self.buckets[bucket_id] if post_id in index]
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(
                f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [
                    len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id])
                                for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(
                    bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert (found_batch or len(self.left_over)
                    >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " +
                  ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()


class EMAModel:
    """
    Maintains (exponential) moving average of a set of parameters.
    Ref: https://github.com/harubaru/waifu-diffusion/diffusers_trainer.py#L478

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from model.parameters()`).
        decay: The exponential decay.
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    @contextlib.contextmanager
    def average_parameters(self, parameters):
        r"""
        Context manager for validation/inference with averaged parameters.
        """
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)
            

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        with_prior_preservation=True,
        size=512,
        center_crop=False,
        num_class_images=None,
        read_prompt_filename=False,
        read_prompt_txt=False,
        append_pos="",
        pad_tokens=False,
        deepdanbooru=False,
        ucg=0,
        debug_prompt=False,
        **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens
        self.deepdanbooru = deepdanbooru
        self.ucg = ucg
        self.debug_prompt = debug_prompt

        self.instance_entries = []
        self.class_entries = []

        if deepdanbooru:
            dd = DeepDanbooru(**kwargs)

        def prompt_resolver(x, default, typ):
            img_item = (x, default)

            if append_pos != typ and append_pos != "both":
                return img_item

            if read_prompt_filename:
                filename = Path(x).stem
                pt = ''.join([i for i in filename if not i.isdigit()])
                pt = pt.replace("_", " ")
                pt = pt.replace("(", "")
                pt = pt.replace(")", "")
                pt = pt.replace("--", "")
                new_prompt = default + " " + pt
                img_item = (x, new_prompt)

            elif read_prompt_txt:
                fp = os.path.splitext(x)[0]
                if not Path(fp + ".txt").is_file() and deepdanbooru:
                    print(f"Deepdanbooru: Working on {x}")
                    return (x, default + dd.get_tags_local(self.read_img(x)))

                with open(fp + ".txt") as f:
                    content = f.read()
                    new_prompt = default + " " + content
                img_item = (x, new_prompt)

            elif deepdanbooru:
                print(f"Deepdanbooru: Working on {x}")
                return (x, default + dd.get_tags_local(self.read_img(x)))

            return img_item

        for concept in concepts_list:
            inst_img_path = [prompt_resolver(x, concept["instance_prompt"], "instance") for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file() and x.suffix != ".txt"]
            self.instance_entries.extend(inst_img_path)

            if with_prior_preservation:
                class_img_path = [prompt_resolver(x, concept["class_prompt"], "class") for x in Path(concept["class_data_dir"]).iterdir() if x.is_file() and x.suffix != ".txt"]
                self.class_entries.extend(class_img_path[:num_class_images])

        if deepdanbooru:
            dd.unload_model()

        random.shuffle(self.instance_entries)
        self.num_instance_images = len(self.instance_entries)
        self.num_class_images = len(self.class_entries)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    @staticmethod
    def read_img(filepath) -> Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def process_tags(tags, min_tags=1, max_tags=32, type_dropout=0.75, keep_important=1.00, keep_jpeg_artifacts=True, sort_tags=False):
        if isinstance(tags, str):
            tags = tags.split(" ")
        final_tags = {}

        tag_dict = {tag: True for tag in tags}
        pure_tag_dict = {tag.split(":", 1)[-1]: tag for tag in tags}
        for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request", "artist_name"]:
            if bad_tag in pure_tag_dict:
                del tag_dict[pure_tag_dict[bad_tag]]

        if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict:
            final_tags["nsfw"] = True

        base_chosen = []
        for tag in tag_dict.keys():
            parts = tag.split(":", 1)
            if parts[0] in ["artist", "copyright", "character"] and random.random() < keep_important:
                base_chosen.append(tag)
            if len(parts[-1]) > 1 and parts[-1][0] in ["1", "2", "3", "4", "5", "6"] and parts[-1][1:] in ["boy", "boys", "girl", "girls"]:
                base_chosen.append(tag)
            if parts[-1] in ["6+girls", "6+boys", "bad_anatomy", "bad_hands"]:
                base_chosen.append(tag)

        tag_count = min(random.randint(min_tags, max_tags), len(tag_dict.keys()))
        base_chosen_set = set(base_chosen)
        chosen_tags = base_chosen + [tag for tag in random.sample(list(tag_dict.keys()), tag_count) if tag not in base_chosen_set]
        if sort_tags:
            chosen_tags = sorted(chosen_tags)

        for tag in chosen_tags:
            tag = tag.replace(",", "").replace("_", " ")
            if random.random() < type_dropout:
                if tag.startswith("artist:"):
                    tag = tag[7:]
                elif tag.startswith("copyright:"):
                    tag = tag[10:]
                elif tag.startswith("character:"):
                    tag = tag[10:]
                elif tag.startswith("general:"):
                    tag = tag[8:]
            if tag.startswith("meta:"):
                tag = tag[5:]
            final_tags[tag] = True

        skip_image = False
        for bad_tag in ["comic", "panels", "everyone", "sample_watermark", "text_focus", "tagme"]:
            if bad_tag in pure_tag_dict:
                skip_image = True
        if not keep_jpeg_artifacts and "jpeg_artifacts" in tag_dict:
            skip_image = True

        return ", ".join(list(final_tags.keys()))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_entries[index % self.num_instance_images]

        if random.random() <= self.ucg:
            instance_prompt = ''

        instance_image = self.read_img(instance_path)
        if self.debug_prompt:
            print(f"instance prompt: {instance_prompt}")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenize(instance_prompt)

        if self.with_prior_preservation:
            class_path, class_prompt = self.class_entries[index % self.num_class_images]
            class_image = self.read_img(class_path)
            if self.debug_prompt:
                print(f"class prompt: {class_prompt}")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenize(class_prompt)

        return example


class AspectRatioDataset(DreamBoothDataset):
    def __init__(self, debug_arb=False, enable_rotate=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug_arb
        self.enable_rotate = enable_rotate
        self.prompt_cache = {}

        # cache prompts for reading
        for path, prompt in self.instance_entries + self.class_entries:
            self.prompt_cache[path] = prompt

    def denormalize(self, img, mean=0.5, std=0.5):
        res = transforms.Normalize((-1*mean/std), (1.0/std))(img.squeeze(0))
        res = torch.clamp(res, 0, 1)
        return res

    def transformer(self, img, size, center_crop=False):
        x, y = img.size
        short, long = (x, y) if x <= y else (y, x)

        w, h = size
        min_crop, max_crop = (w, h) if w <= h else (h, w)
        ratio_src, ratio_dst = float(long / short), float(max_crop / min_crop)
        
        if (x>y and w<h) or (x<y and w>h) and self.with_prior_preservation and self.enable_rotate:
            # handle i/c mixed input
            img = img.rotate(90, expand=True)
            x, y = img.size
            
        if ratio_src > ratio_dst:
            new_w, new_h = (min_crop, int(min_crop * ratio_src)) if x<y else (int(min_crop * ratio_src), min_crop)
        elif ratio_src < ratio_dst:
            new_w, new_h = (max_crop, int(max_crop / ratio_src)) if x>y else (int(max_crop / ratio_src), max_crop)
        else:
            new_w, new_h = w, h

        image_transforms = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((h, w)) if center_crop else transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        new_img = image_transforms(img)

        if self.debug:
            import uuid, torchvision
            print(x, y, "->", new_w, new_h, "->", new_img.shape)
            filename = str(uuid.uuid4())
            torchvision.utils.save_image(self.denormalize(new_img), f"/tmp/{filename}_1.jpg")
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(img), f"/tmp/{filename}_2.jpg")
            print(f"saved: /tmp/{filename}")

        return new_img

    def build_dict(self, item_id, size, typ) -> dict:
        if item_id == "":
            return {}
        prompt = self.prompt_cache[item_id]
        image = self.read_img(item_id)
        
        if random.random() < self.ucg:
            prompt = ''
            
        if self.debug_prompt:
            print(f"{typ} prompt: {prompt}")
        
        example = {
            f"{typ}_images": self.transformer(image, size),
            f"{typ}_prompt_ids": self.tokenize(prompt)
        }
        return example

    def __getitem__(self, index):
        result = []
        for item in index:
            instance_dict = self.build_dict(item["instance"], item["size"], "instance")
            class_dict = self.build_dict(item["class"], item["size"], "class")
            result.append({**instance_dict, **class_dict})

        return result


class AspectRatioSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        instance_buckets: AspectRatioBucket, 
        class_buckets: AspectRatioBucket, 
        num_replicas: int = 1,
        with_prior_preservation: bool = False,
        debug: bool = False,
    ):
        super().__init__(None)
        self.instance_bucket_manager = instance_buckets
        self.class_bucket_manager = class_buckets 
        self.num_replicas = num_replicas
        self.debug = debug
        self.with_prior_preservation = with_prior_preservation
        self.iterator = instance_buckets if len(class_buckets) < len(instance_buckets) or \
            not with_prior_preservation else class_buckets
            
    def build_res_id_dict(self, iter):
        base = {}
        for item, res in iter.generator():
            base.setdefault(res,[]).extend([item[0]])
        return base

    def find_closest(self, size, size_id_dict, typ):
        new_size = size
        if size not in size_id_dict or not any(size_id_dict[size]):
            kv = [(abs(s[0] / s[1] - size[0] / size[1]), s) for s in size_id_dict.keys() if any(size_id_dict[s])]            
            kv.sort(key=lambda e: e[0])
                
            new_size = kv[0][1]
            print(f"Warning: no {typ} image with {size} exists. Will use the closest ratio {new_size}.")

        return random.choice(size_id_dict[new_size])
    
    def __iter__(self):
        iter_is_instance = self.iterator == self.instance_bucket_manager
        self.cached_ids = self.build_res_id_dict(self.class_bucket_manager if iter_is_instance else self.instance_bucket_manager)   
        
        
        for batch, size in self.iterator.generator():
            result = []
            
            for item in batch:
                sdict = {"size": size}
                
                if iter_is_instance:
                    rdict = {"instance": item, "class": self.find_closest(size, self.cached_ids, "class") if self.with_prior_preservation else ""}
                else:
                    rdict = {"class": item, "instance": self.find_closest(size, self.cached_ids, "instance")}
                    
                    
                result.append({**rdict, **sdict})

            yield result
            
    def __len__(self):
        return len(self.iterator) // self.num_replicas


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer_class(optimizer_name: str):
    def try_import_bnb():
        try:
            import bitsandbytes as bnb
            return bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit optimizers, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
    def try_import_ds():
        try:
            import deepspeed
            return deepspeed
        except ImportError:
            raise ImportError(
                 "Failed to import Deepspeed"
             )

    name = optimizer_name.lower()

    if name == "adamw":
        return torch.optim.AdamW
    elif name == "adamw_8bit":
        return try_import_bnb().optim.AdamW8bit
    elif name == "adamw_ds":
        return try_import_ds().ops.adam.DeepSpeedCPUAdam
    elif name == "sgdm":
        return torch.optim.sgd
    elif name == "sgdm_8bit":
        return try_import_bnb().optim.SGD8bit
    else:
        raise ValueError("Unsupport optimizer")


def generate_class_images(args, accelerator):
    pipeline = None
    for concept in args.concepts_list:
        class_images_dir = Path(concept["class_data_dir"])
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if pipeline is None:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path, subfolder=None if args.pretrained_vae_name_or_path else "vae"),
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                )
                pipeline.set_progress_bar_config(disable=True)
                pipeline.to(accelerator.device)

            num_new_images = args.num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset([concept["class_prompt"], concept["class_negative_prompt"]], num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            sample_dataloader = accelerator.prepare(sample_dataloader)

            with torch.autocast("cuda"), torch.inference_mode():
                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(prompt=example["prompt"][0][0],
                                      negative_prompt=example["prompt"][1][0],
                                      guidance_scale=args.save_guidance_scale,
                                      num_inference_steps=args.save_infer_steps,
                                      num_images_per_prompt=len(example["prompt"][0])).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()              
       
                    
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.
    :return:
    """
    devid = torch.cuda.current_device()
    return f"GPU.{devid} {torch.cuda.get_device_name(devid)}"


def init_arb_buckets(args, accelerator):
    arg_config = {
        "bsz": args.train_batch_size,
        "seed": args.seed,
        "debug": args.debug_arb,
        "base_res": (args.resolution, args.resolution),
        "max_size": args.arb_max_size,
        "divisible": args.arb_divisible,
        "max_ar_error": args.arb_max_ar_error,
        "min_dim": args.arb_min_dim,
        "dim_limit": args.arb_dim_limit,
        "world_size": accelerator.num_processes,
        "global_rank": args.local_rank,
    }

    if args.debug_arb:
        print("BucketManager initialized using config:")
        print(json.dumps(arg_config, sort_keys=True, indent=4))
    else:
        print(f"BucketManager initialized with base_res = {arg_config['base_res']}, max_size = {arg_config['max_size']}")
        
    def get_id_size_dict(entries, hint):
        id_size_map = {}

        for entry in tqdm(entries, desc=f"Loading resolution from {hint} images", disable=args.local_rank not in [0, -1]):
            with Image.open(entry) as img:
                size = img.size
            id_size_map[entry] = size

        return id_size_map
    
    instance_entries, class_entries  = [], []
    for concept in args.concepts_list:
        inst_img_path = [x for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file() and x.suffix != ".txt"]
        instance_entries.extend(inst_img_path)
        
        if args.with_prior_preservation:
            class_img_path = [x for x in Path(concept["class_data_dir"]).iterdir() if x.is_file() and x.suffix != ".txt"]
            class_entries.extend(class_img_path[:args.num_class_images])
            
    instance_id_size_map = get_id_size_dict(instance_entries, "instance")
    class_id_size_map = get_id_size_dict(class_entries, "class")

    instance_bucket_manager = AspectRatioBucket(instance_id_size_map, **arg_config)
    class_bucket_manager = AspectRatioBucket(class_id_size_map, **arg_config)
    
    return instance_bucket_manager, class_bucket_manager


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    metrics = ["tensorboard"]
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_name, reinit=False)
        metrics.append("wandb")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=metrics,
        logging_dir=logging_dir,
    )
    
    print(get_gpu_ram())
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "class_negative_prompt": args.class_negative_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        if type(args.concepts_list) == str:
            with open(args.concepts_list, "r") as f:
                args.concepts_list = json.load(f)

    if args.with_prior_preservation and accelerator.is_local_main_process:
        generate_class_images(args, accelerator)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer")
    else:
        raise ValueError(args.tokenizer_name)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    def encode_tokens(tokens):

        if args.clip_skip > 1:
            result = text_encoder(tokens, output_hidden_states=True, return_dict=True)
            return text_encoder.text_model.final_layer_norm(result.hidden_states[-args.clip_skip])

        return text_encoder(tokens)[0]

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.to(torch.float32)

    if args.xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )

    elif args.scale_lr_sqrt:
        args.learning_rate *= math.sqrt(args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    elif args.use_deepspeed_adam:
        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "Failed to import Deepspeed"
            )
        optimizer_class = deepspeed.ops.adam.DeepSpeedCPUAdam
    else:
        optimizer_class = get_optimizer_class(args.optimizer)

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    if "adam" in args.optimizer.lower():
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    elif "sgd" in args.optimizer.lower():
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            dampening=args.sgd_dampening,
            weight_decay=args.weight_decay
         )
    else:
        raise ValueError(args.optimizer)

    noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    dataset_class = AspectRatioDataset if args.use_aspect_ratio_bucket else DreamBoothDataset
    train_dataset = dataset_class(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        read_prompt_filename=args.read_prompt_filename,
        read_prompt_txt=args.read_prompt_txt,
        append_pos=args.append_prompt,
        bsz=args.train_batch_size,
        debug_arb=args.debug_arb,
        seed=args.seed,
        deepdanbooru=args.deepdanbooru,
        dd_threshold=args.dd_threshold,
        dd_alpha_sort=args.dd_alpha_sort,
        dd_use_spaces=args.dd_use_spaces,
        dd_use_escape=args.dd_use_escape,
        dd_include_ranks=args.dd_include_ranks,
        enable_rotate=args.enable_rotate,
        ucg=args.ucg,
        debug_prompt=args.debug_prompt,
    )
    
    def collate_fn_wrap(examples):
        # workround for variable list
        if len(examples) == 1:
            examples = examples[0]
        return collate_fn(examples)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    if args.ucg:
        args.not_cache_latents = True
        print("Latents cache disabled.")

    if args.use_aspect_ratio_bucket:
        args.not_cache_latents = True
        print("Latents cache disabled.")
        instance_bucket_manager, class_bucket_manager = init_arb_buckets(args, accelerator)
        sampler = AspectRatioSampler(instance_bucket_manager, class_bucket_manager, accelerator.num_processes, args.with_prior_preservation)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=collate_fn_wrap, num_workers=1, sampler=sampler,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=1
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(encode_tokens(batch["input_ids"]))

        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler == "cosine_with_restarts_mod":
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.num_cycles,
            last_epoch=args.last_epoch,
        )
    elif args.lr_scheduler == "cosine_mod":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.num_cycles,
            last_epoch=args.last_epoch,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    base_step = 0
    base_epoch = 0

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        
    if args.resume:
        state_dict = torch.load(os.path.join(args.pretrained_model_name_or_path, f"state.pt"), map_location="cuda")
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

        if "scheduler" in state_dict and not args.reinit_scheduler:
            lr_scheduler.load_state_dict(state_dict["scheduler"])

            last_lr = state_dict["scheduler"]["_last_lr"]
            print(f"Loaded state_dict from '{args.pretrained_model_name_or_path}': last_lr = {last_lr}")

        base_step = state_dict["total_steps"]
        base_epoch = state_dict["total_epoch"]
        del state_dict

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(interrupt=False):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:

            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

            unet_unwrapped = accelerator.unwrap_model(unet)
                
            if args.save_unet_half or args.unet_half:
                import copy
                unet_unwrapped = copy.deepcopy(unet_unwrapped).half()

            scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet_unwrapped,
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path, subfolder=None if args.pretrained_vae_name_or_path else "vae"),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=weight_dtype,
            )

            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            save_dir = output_dir / f"checkpoint_{global_step}" 
            if local_step >= args.max_train_steps:
                save_dir = output_dir / f"checkpoint_last"

            save_dir.mkdir(exist_ok=True)
            pipeline.save_pretrained(save_dir)
            print(f"[*] Weights saved at {save_dir}")

            if args.use_ema:
                ema_path = save_dir / "unet_ema"

                ema_unet.store(unet_unwrapped.parameters())
                ema_unet.copy_to(unet_unwrapped.parameters())

                # with ema_unet.average_parameters(unet_unwrapped.parameters()):
                try:
                    unet_unwrapped.save_pretrained(ema_path)
                finally:
                    ema_unet.restore(unet_unwrapped.parameters())
            
                ema_unet.to("cpu", dtype=weight_dtype)
                torch.cuda.empty_cache()
                print(f"[*] EMA Weights saved at {ema_path}")

            if args.save_states:
                accelerator.save({
                    'total_epoch': global_epoch,
                    'total_steps': global_step,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'loss': loss,
                }, os.path.join(save_dir, "state.pt"))

            with open(save_dir / "args.json", "w") as f:
                args.resume_from = str(save_dir)
                json.dump(args.__dict__, f, indent=2)

            if interrupt:
                return

            if args.save_sample_prompt:
                pipeline = pipeline.to(accelerator.device)
                g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                pipeline.set_progress_bar_config(disable=True)
                sample_dir = save_dir / "samples"
                sample_dir.mkdir(exist_ok=True)
                with torch.autocast("cuda"), torch.inference_mode():
                    for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(sample_dir / f"{i}.png")

                if args.wandb:
                    wandb.log({"samples": [wandb.Image(str(x)) for x in sample_dir.glob("*.png")]}, step=global_step)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if args.use_ema:
                ema_unet.to(accelerator.device, dtype=weight_dtype)

            if args.wandb_artifact:
                model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
                    'epochs_trained': global_epoch + 1,
                    'project': run.project
                })
                model_artifact.add_dir(save_dir)
                wandb.log_artifact(model_artifact, aliases=['latest', 'last', f'epoch {global_epoch + 1}'])

                if args.rm_after_wandb_saved:
                    shutil.rmtree(save_dir)
                    subprocess.run("wandb", "artifact", "cache", "cleanup", "1G")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    local_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()

    @atexit.register
    def on_exit():
        if 100 < local_step < args.max_train_steps and accelerator.is_local_main_process:
            print("Saving model...")
            save_weights(interrupt=True)

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = encode_tokens(batch[0][1])
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = encode_tokens(batch["input_ids"])

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_avg.update(loss.detach_(), bsz)

            global_step = base_step + local_step
            global_epoch = base_epoch + epoch

            if not local_step % args.log_interval:
                logs = {
                    "epoch": global_epoch + 1,
                    "loss": loss_avg.avg.item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                local_step += 1
                
            if local_step > args.save_min_steps and not global_step % args.save_interval:
                save_weights()

            if local_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    save_weights()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)