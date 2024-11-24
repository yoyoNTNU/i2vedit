import argparse
import os
import platform
import re
import warnings
import imageio
import random
from typing import Optional
from tqdm import trange
from einops import rearrange

import torch
from torch import Tensor
from torch.nn.functional import interpolate
from diffusers import StableVideoDiffusionPipeline, EulerDiscreteScheduler 
from diffusers import TextToVideoSDPipeline

from i2vedit.train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from i2vedit.utils.lora_handler import LoraHandler
from i2vedit.utils.model_utils import P2PStableVideoDiffusionPipeline


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
    load_spatial_lora: bool = False,
    dtype = torch.float16
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, feature_extractor, image_encoder, vae, unet = load_primary_models(model)

    # Freeze any necessary models
    freeze_models([vae, image_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(xformers, sdp, unet)

    lora_manager_temporal = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        use_image_lora=False,
        save_for_webui=False,
        only_for_webui=False,
        unet_replace_modules=["TemporalBasicTransformerBlock"],
        image_encoder_replace_modules=None,
        lora_bias=None
    )

    unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
        True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale)

    if load_spatial_lora:
        lora_manager_spatial = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=True,
            use_image_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            unet_replace_modules=["BasicTransformerBlock"],
            image_encoder_replace_modules=None,
            lora_bias=None
        )

        spatial_lora_path = lora_path.replace("temporal", "spatial") 
        unet_lora_params, unet_negation = lora_manager_spatial.add_lora_to_model(
            True, unet, lora_manager_spatial.unet_replace_modules, 0, spatial_lora_path, r=lora_rank, scale=lora_scale)

    unet.eval()
    image_encoder.eval()
    unet_and_text_g_c(unet, image_encoder, False, False)

    pipe = P2PStableVideoDiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder.to(device=device, dtype=dtype),
        vae=vae.to(device=device, dtype=dtype),
        unet=unet.to(device=device, dtype=dtype)
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe
