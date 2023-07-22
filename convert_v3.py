import os
import argparse
from io import BytesIO
from typing import Optional
import safetensors.torch

from omegaconf import OmegaConf
import requests
import torch
from transformers import (
    CLIPTextModel,
    CLIPTextConfig,
    CLIPTokenizer
)
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config
)


def load_model(path):
    if path.endswith(".safetensors"):
        m = safetensors.torch.load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    return state_dict


def convert_to_df(checkpoint, config_path="./v1-inference.yaml", return_pipe=False, extract_ema=False):
    # key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
    # key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
    # key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"

    global_step = None
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]

    # model_type = "v1"
    # config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    upcast_attention = None
    # if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
    #     # model_type = "v2"
    #     config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"

    #     if global_step == 110000:
    #         # v2.1 needs to upcast attention
    #         upcast_attention = True
    # elif key_name_sd_xl_base in checkpoint:
    #     # only base xl has two text embedders
    #     config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    # elif key_name_sd_xl_refiner in checkpoint:
    #     # only refiner xl has embedder and one text embedders
    #     config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

    # original_config_file = BytesIO(requests.get(config_url).content)
    original_config_file = BytesIO(open(config_path, "rb").read())
    original_config = OmegaConf.load(original_config_file)

    # Convert the text model.
    if (
        "cond_stage_config" in original_config.model.params
        and original_config.model.params.cond_stage_config is not None
    ):
        model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    elif original_config.model.params.network_config is not None:
        if original_config.model.params.network_config.params.context_dim == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        prediction_type = "epsilon"
        image_size = 512

    num_train_timesteps = getattr(original_config.model.params, "timesteps", None) or 1000
    beta_start = getattr(original_config.model.params, "linear_start", None) or 0.02
    beta_end = getattr(original_config.model.params, "linear_end", None) or 0.085
    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention
    unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, extract_ema=extract_ema)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    if model_type == "FrozenOpenCLIPEmbedder":
        text_model = convert_open_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    elif model_type == "FrozenCLIPEmbedder":
        keys = list(checkpoint.keys())
        text_model_dict = {}
        for key in keys:
            if key.startswith("cond_stage_model.transformer"):
                dest_key = key[len("cond_stage_model.transformer."):]
                if "text_model" not in dest_key:
                    dest_key = f"text_model.{dest_key}"
                text_model_dict[dest_key] = checkpoint[key]

        text_model = CLIPTextModel(CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14"))
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if "text_model.embeddings.position_ids" not in text_model.state_dict().keys() \
                and "text_model.embeddings.position_ids" in text_model_dict.keys():
            del text_model_dict["text_model.embeddings.position_ids"]

        if len(text_model_dict) < 10:
            text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    if not return_pipe:
        return converted_unet_checkpoint, converted_vae_checkpoint, text_model_dict
    else:
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
        unet.load_state_dict(converted_unet_checkpoint)
        text_model.load_state_dict(text_model_dict)
        pipe = StableDiffusionPipeline(
            unet=unet,
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        default=False,
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    # parser.add_argument(
    #     "--vae_path", default=None, type=str, help="Path to the vae to convert."
    # )
    # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    if args.original_config_file is None:
        if not os.path.exists("./v1-inference.yaml"):
            os.system(
                "wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
            )
        args.original_config_file = "./v1-inference.yaml"

    pipe = convert_to_df(load_model(args.checkpoint_path), config_path=args.original_config_file, return_pipe=True, extract_ema=args.extract_ema)
    pipe.save_pretrained(args.dump_path)
