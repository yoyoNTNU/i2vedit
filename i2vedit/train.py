import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
from scipy.stats import anderson
import imageio
import numpy as np

from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import diffusers
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.unet_3d_blocks import \
            (CrossAttnDownBlockSpatioTemporal, 
             DownBlockSpatioTemporal,
             CrossAttnUpBlockSpatioTemporal,
             UpBlockSpatioTemporal)

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPEncoder

from i2vedit.utils.dataset import (
    CachedDataset,
)

from i2vedit.utils.lora_handler import LoraHandler
from i2vedit.utils.lora import extract_lora_child_module
from i2vedit.utils.euler_utils import euler_inversion
from i2vedit.utils.svd_util import SmoothAreaRandomDetection
from i2vedit.utils.model_utils import (
    tensor_to_vae_latent,
    P2PEulerDiscreteScheduler,
    P2PStableVideoDiffusionPipeline
)
from i2vedit.data import ResolutionControl, SingleClipDataset
from i2vedit.prompt_attention import attention_util

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)


def export_to_video(video_frames, output_video_path, fps, resctrl:ResolutionControl):
    flattened_video_frames = [img for sublist in video_frames for img in sublist]
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in flattened_video_frames:
        img = resctrl.callback(img)
        video_writer.append_data(np.array(img))
    video_writer.close()


def create_output_folders(output_dir, config, clip_id):
    out_dir = os.path.join(output_dir, f"train_motion_lora/clip_{clip_id}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    # OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def load_primary_models(pretrained_model_path):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_path, subfolder="feature_extractor", revision=None
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_path, subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_path, subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    return noise_scheduler, feature_extractor, image_encoder, vae, unet


def unet_and_text_g_c(unet, image_encoder, unet_enable, image_enable):
    unet.gradient_checkpointing = unet_enable
    unet.mid_block.gradient_checkpointing = unet_enable
    for module in unet.down_blocks + unet.up_blocks:
        if isinstance(module, 
            (CrossAttnDownBlockSpatioTemporal, 
             DownBlockSpatioTemporal,
             CrossAttnUpBlockSpatioTemporal,
             UpBlockSpatioTemporal)):
            module.gradient_checkpointing = unet_enable


def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)


def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])


def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }


def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW


def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)


def inverse_video(pipe, latents, num_steps, image):
    euler_inv_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    euler_inv_scheduler.set_timesteps(num_steps)

    euler_inv_latent = euler_inversion(
        pipe, euler_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, image=image)[-1]
    return euler_inv_latent


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        cached_latent_dir=None,
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float32)
    #vae.enable_slicing()

    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None
    )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float32)
            refer_pixel_values = batch['refer_pixel_values'].to('cuda', dtype=torch.float32)
            cross_pixel_values = batch['cross_pixel_values'].to('cuda', dtype=torch.float32)
            batch['latents'] = tensor_to_vae_latent(pixel_values, vae)
            if batch.get("refer_latents") is None:
                batch['refer_latents'] = tensor_to_vae_latent(refer_pixel_values, vae)
            batch['cross_latents'] = tensor_to_vae_latent(cross_pixel_values, vae)
                
            for k, v in batch.items(): batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break

                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params += 1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
            alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def should_sample(global_step, validation_steps, validation_data):
    return global_step % validation_steps == 0 and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        image_encoder,
        vae,
        output_dir,
        lora_manager_spatial: LoraHandler,
        lora_manager_temporal: LoraHandler,
        unet_target_replace_module=None,
        image_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True,
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, i_dtype, v_dtype = unet.dtype, image_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet.cpu(), keep_fp32_wrapper=False))
    image_encoder_out = copy.deepcopy(accelerator.unwrap_model(image_encoder.cpu(), keep_fp32_wrapper=False))
    pipeline = P2PStableVideoDiffusionPipeline.from_pretrained(
        path,
        unet=unet_out,
        image_encoder=image_encoder_out,
        vae=accelerator.unwrap_model(vae),
#        torch_dtype=weight_dtype,
    ).to(torch_dtype=torch.float32)

#    lora_manager_spatial.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/spatial', step=global_step)
    if lora_manager_temporal is not None:
        lora_manager_temporal.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/temporal', step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, image_encoder = accelerator.prepare(unet, image_encoder)
        models_to_cast_back = [(unet, u_dtype), (image_encoder, i_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del image_encoder_out
    torch.cuda.empty_cache()
    gc.collect()

def load_images_from_list(img_list):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split('_')
        if len(parts) > 1 and parts[0] == 'frame':
            try:
                return int(parts[1].split('.')[0])  # Extracting the number part
            except ValueError:
                return float('inf')  # In case of non-integer part, place this file at the end
        return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number
    #sorted_files = sorted(os.listdir(folder), key=frame_number)
    sorted_files = img_list

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(filename).convert('RGB')
            images.append(img)

    return images

# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data, u


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def train_motion_lora(
        pretrained_model_path,
        output_dir: str,
        train_dataset: SingleClipDataset,
        validation_data: Dict,
        edited_firstframes: List[Image.Image],
        train_data: Dict,
        validation_images: List[Image.Image],
        validation_images_latents: List[torch.Tensor],
        clip_id: int,
        consistency_controller: attention_util.ConsistencyAttentionControl = None, 
        consistency_edit_controller_list: List[attention_util.ConsistencyAttentionControl] = [None,],
        consistency_find_modules: Dict = {},
        single_spatial_lora: bool = False,
        train_temporal_lora: bool = True,
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = None,  # Eg: ("attn1", "attn2")
        extra_unet_params=None,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 5e-5,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        image_encoder_gradient_checkpointing: bool = False,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        resume_step: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,
        extend_dataset: bool = False,
        cache_latents: bool = False,
        cached_latent_dir=None,
        use_unet_lora: bool = False,
        unet_lora_modules: Tuple[str] = [],
        image_encoder_lora_modules: Tuple[str] = [],
        save_pretrained_model: bool = True,
        lora_rank: int = 16,
        lora_path: str = '',
        lora_unet_dropout: float = 0.1,
        logger_type: str = 'tensorboard',
        **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # Handle the output folder creation
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config, clip_id)
    
    # Load scheduler, tokenizer and models.
    noise_scheduler, feature_extractor, image_encoder, vae, unet = load_primary_models(pretrained_model_path)

    # Freeze any necessary models
    freeze_models([vae, image_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)
    
    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    #extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    #extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    # Temporal LoRA
    if train_temporal_lora:
        # one temporal lora
        lora_manager_temporal = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["TemporalBasicTransformerBlock"])

        unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
            use_unet_lora, unet, lora_manager_temporal.unet_replace_modules, lora_unet_dropout,
            lora_path + '/temporal/lora/', r=lora_rank)

        optimizer_temporal = optimizer_cls(
            create_optimizer_params([param_optim(unet_lora_params_temporal, use_unet_lora, is_lora=True,
                                                 extra_params={**{"lr": learning_rate}}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        lr_scheduler_temporal = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_temporal,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
    else:
        lora_manager_temporal = None
        unet_lora_params_temporal, unet_negation_temporal = [], []
        optimizer_temporal = None
        lr_scheduler_temporal = None

    # Spatial LoRAs
    if single_spatial_lora:
        spatial_lora_num = 1
    else:
        # one spatial lora for each video
        spatial_lora_num = train_dataset.__len__()

    lora_managers_spatial = []
    unet_lora_params_spatial_list = []
    optimizer_spatial_list = []
    lr_scheduler_spatial_list = []
    for i in range(spatial_lora_num):
        lora_manager_spatial = LoraHandler(use_unet_lora=use_unet_lora, unet_replace_modules=["BasicTransformerBlock"])
        lora_managers_spatial.append(lora_manager_spatial)
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            use_unet_lora, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
            lora_path + '/spatial/lora/', r=lora_rank)

        unet_lora_params_spatial_list.append(unet_lora_params_spatial)

        optimizer_spatial = optimizer_cls(
            create_optimizer_params([param_optim(unet_lora_params_spatial, use_unet_lora, is_lora=True,
                                                 extra_params={**{"lr": learning_rate}}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        optimizer_spatial_list.append(optimizer_spatial)

        # Scheduler
        lr_scheduler_spatial = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_spatial,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        lr_scheduler_spatial_list.append(lr_scheduler_spatial)

        unet_negation_all = unet_negation_spatial + unet_negation_temporal

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        cached_latent_dir
    )

    if cached_data_loader is not None and train_data.get("use_data_aug") is None:
        train_dataloader = cached_data_loader

    # Prepare everything with our `accelerator`.
    unet, optimizer_temporal, train_dataloader, lr_scheduler_temporal, image_encoder = accelerator.prepare(
        unet,
        optimizer_temporal,
        train_dataloader,
        lr_scheduler_temporal,
        image_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        image_encoder,
        gradient_checkpointing,
        image_encoder_gradient_checkpointing
    )

    # Enable VAE slicing to save memory.
    #vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [image_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # Fix noise schedules to predcit light and dark areas if available.
    # if not use_offset_noise and rescale_schedule:
    #    noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("svd-finetune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training for motion lora*****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
#        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings= image_embeddings.unsqueeze(1)
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
        device=None,  # Add a device parameter
    ):
        # Determine the target device
        target_device = device if device is not None else 'cpu'
    
        # Ensure motion_bucket_ids is a tensor and on the target device
        if not isinstance(motion_bucket_ids, torch.Tensor):
            motion_bucket_ids = torch.tensor(motion_bucket_ids, dtype=dtype, device=target_device)
        else:
            motion_bucket_ids = motion_bucket_ids.to(device=target_device)
    
        # Reshape motion_bucket_ids if necessary
        if motion_bucket_ids.dim() == 1:
            motion_bucket_ids = motion_bucket_ids.view(-1, 1)
    
        # Check for batch size consistency
        if motion_bucket_ids.size(0) != batch_size:
            raise ValueError("The length of motion_bucket_ids must match the batch_size.")
    
        # Create fps and noise_aug_strength tensors on the target device
        add_time_ids = torch.tensor([fps, noise_aug_strength], dtype=dtype, device=target_device).repeat(batch_size, 1)
    
        # Concatenate with motion_bucket_ids
        add_time_ids = torch.cat([add_time_ids, motion_bucket_ids], dim=1)
    
        # Checking the dimensions of the added time embedding
        passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(1)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
                "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
    
        return add_time_ids

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # set consistency controller
    if consistency_controller is not None:
        consistency_train_controller = attention_util.ConsistencyAttentionControl(
             additional_attention_store=consistency_controller,
             use_inversion_attention=True,
             save_self_attention=False,
             save_latents=False,
             disk_store=True
        )
        attention_util.register_attention_control(
            unet,
            None,
            consistency_train_controller,
            find_modules={},
            consistency_find_modules=consistency_find_modules
        )

    def finetune_unet(batch, step, mask_spatial_lora=False, mask_temporal_lora=False):
        nonlocal use_offset_noise
        nonlocal rescale_schedule


        # Unfreeze UNET Layers
        if global_step == 0:
            already_printed_trainables = False
            unet.train()
            handle_trainable_modules(
                unet,
                trainable_modules,
                is_enabled=True,
                negation=unet_negation_all
            )

        # Convert videos to latent space
        #print("use_data_aug", train_data.get("use_data_aug"))
        if not cache_latents or train_data.get("use_data_aug") is not None:
            latents = tensor_to_vae_latent(batch["pixel_values"], vae)
            refer_latents = tensor_to_vae_latent(batch["refer_pixel_values"], vae)
            cross_latents = tensor_to_vae_latent(batch["cross_pixel_values"], vae)
        else:
            latents = batch["latents"]
            refer_latents = batch["refer_latents"]
            cross_latents = batch["cross_latents"]

        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        noise_1 = sample_noise(latents, offset_noise_strength, False)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        sigmas, u = rand_cosine_interpolated(shape=[bsz,], image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                          sigma_data=sigma_data, min_value=min_value, max_value=max_value)
        noise_scheduler.set_timesteps(validation_data.num_inference_steps, device=latents.device)
        all_sigmas = noise_scheduler.sigmas
        sigmas = sigmas.to(latents.device)
        timestep = (validation_data.num_inference_steps - torch.searchsorted(all_sigmas.to(latents.device).flip(dims=(0,)), sigmas, right=False)).clamp(0,validation_data.num_inference_steps-1)[0]
        u = u.item()
        if consistency_controller is not None:
            #timestep = int(u * (validation_data.num_inference_steps-1)+0.5)
            #print("u", u, "timestep", timestep, "sigmas", sigmas, "all_sigmas", all_sigmas)
            consistency_train_controller.set_cur_step(timestep)
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas_reshaped = sigmas.clone()
        while len(sigmas_reshaped.shape) < len(latents.shape):
            sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
        
        # add noise to the latents or the original image?
        train_noise_aug = 0.02
        conditional_latents = refer_latents / vae.config.scaling_factor
        small_noise_latents = conditional_latents + noise_1[:,0:1,:,:,:] * train_noise_aug
        conditional_latents = small_noise_latents[:, 0, :, :, :]

        noisy_latents = latents + noise * sigmas_reshaped

        timesteps = torch.Tensor(
            [0.25 * sigma.log() for sigma in sigmas]).to(latents.device)
        
        inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)

        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()
            image_encoder.eval()

        # Get the text embedding for conditioning.
        encoder_hidden_states = encode_image(
            batch["cross_pixel_values"][:, 0, :, :, :])
        detached_encoder_state = encoder_hidden_states.clone().detach()
        
        added_time_ids = _get_add_time_ids(
            6,
            batch["motion_values"],
            train_noise_aug, # noise_aug_strength == 0.0
            encoder_hidden_states.dtype,
            bsz,
            unet,
            device=latents.device
        )
        added_time_ids = added_time_ids.to(latents.device)

        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        conditioning_dropout_prob = kwargs.get('conditioning_dropout_prob')
        if conditioning_dropout_prob is not None:
            random_p = torch.rand(
                bsz, device=latents.device, generator=generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.where(
                prompt_mask, null_conditioning, encoder_hidden_states)

            # Sample masks for the original images.
            image_mask_dtype = conditional_latents.dtype
            image_mask = 1 - (
                (random_p >= conditioning_dropout_prob).to(
                    image_mask_dtype)
                * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            conditional_latents = image_mask * conditional_latents

        # Concatenate the `conditional_latents` with the `noisy_latents`.
        conditional_latents = conditional_latents.unsqueeze(
            1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
        inp_noisy_latents = torch.cat(
            [inp_noisy_latents, conditional_latents], dim=2)

        # Get the target for loss depending on the prediction type
        # if noise_scheduler.config.prediction_type == "epsilon":
        #     target = latents  # we are computing loss against denoise latents
        # elif noise_scheduler.config.prediction_type == "v_prediction":
        #     target = noise_scheduler.get_velocity(
        #         latents, noise, timesteps)
        # else:
        #     raise ValueError(
        #         f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        target = latents

        encoder_hidden_states = detached_encoder_state

        if True:#mask_spatial_lora:
            loras = extract_lora_child_module(unet, target_replace_module=["BasicTransformerBlock"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_spatial = None
        else:
            loras = extract_lora_child_module(unet, target_replace_module=["BasicTransformerBlock"])

            if spatial_lora_num == 1:
                for lora_i in loras:
                    lora_i.scale = 1.
            else:
                for lora_i in loras:
                    lora_i.scale = 0.

                for lora_idx in range(0, len(loras), spatial_lora_num):
                    loras[lora_idx + step].scale = 1.

            loras = extract_lora_child_module(unet, target_replace_module=["TemporalBasicTransformerBlock"])
            if len(loras) > 0:
                for lora_i in loras:
                    lora_i.scale = 0.

            ran_idx = 0#torch.randint(0, noisy_latents.shape[2], (1,)).item()

            #spatial_inp_noisy_latents = inp_noisy_refer_latents[:, ran_idx:ran_idx+1, :, :, :]
            inp_noisy_spatial_latents = inp_noisy_latents#[:, ran_idx:ran_idx+1, :, :, :]

            target_spatial = latents#[:, ran_idx:ran_idx+1, :, :, :]
            # Predict the noise residual
            model_pred = unet(
                inp_noisy_spatial_latents, timesteps, encoder_hidden_states,
                added_time_ids
            ).sample

            sigmas = sigmas_reshaped
            # Denoise the latents
            c_out = -sigmas / ((sigmas**2 + 1)**0.5)
            c_skip = 1 / (sigmas**2 + 1)
            denoised_latents = model_pred * c_out + c_skip * noisy_latents#[:, ran_idx:ran_idx+1, :, :, :]
            weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

            # MSE loss
            loss_spatial = torch.mean(
                (weighing.float() * (denoised_latents.float() -
                 target_spatial.float()) ** 2).reshape(target_spatial.shape[0], -1),
                dim=1,
            )
            loss_spatial = loss_spatial.mean()

        if mask_temporal_lora:
            loras = extract_lora_child_module(unet, target_replace_module=["TemporalBasicTransformerBlock"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_temporal = None
        else:
            loras = extract_lora_child_module(unet, target_replace_module=["TemporalBasicTransformerBlock"])
            for lora_i in loras:
                lora_i.scale = 1.
            # Predict the noise residual
            model_pred = unet(
                inp_noisy_latents, timesteps, encoder_hidden_states,
                added_time_ids=added_time_ids,
            ).sample

            sigmas = sigmas_reshaped
            # Denoise the latents
            c_out = -sigmas / ((sigmas**2 + 1)**0.5)
            c_skip = 1 / (sigmas**2 + 1)
            denoised_latents = model_pred * c_out + c_skip * noisy_latents
            if consistency_controller is not None:
                consistency_train_controller.step_callback(denoised_latents.detach())
            weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

            # MSE loss
            loss_temporal = torch.mean(
                (weighing.float() * (denoised_latents.float() -
                 target.float()) ** 2).reshape(target.shape[0], -1),
                dim=1,
            )
            loss_temporal = loss_temporal.mean()

#            beta = 1
#            alpha = (beta ** 2 + 1) ** 0.5
#            ran_idx = torch.randint(0, model_pred.shape[1], (1,)).item()
#            model_pred_decent = alpha * model_pred - beta * model_pred[:, ran_idx, :, :, :].unsqueeze(1)
#            target_decent = alpha * target - beta * target[:, ran_idx, :, :, :].unsqueeze(1)
#            loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
            loss_temporal = loss_temporal #+ loss_ad_temporal

        return loss_spatial, loss_temporal, latents, noise

    for epoch in range(first_epoch, num_train_epochs):
        train_loss_spatial = 0.0
        train_loss_temporal = 0.0
        

        for step, batch in enumerate(train_dataloader):
            #torch.cuda.empty_cache() 
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):

                for optimizer_spatial in optimizer_spatial_list:
                    optimizer_spatial.zero_grad(set_to_none=True)

                if optimizer_temporal is not None:
                    optimizer_temporal.zero_grad(set_to_none=True)

                if train_temporal_lora:
                    mask_temporal_lora = False
                else:
                    mask_temporal_lora = True
                if False:#clip_id != 0:
                    mask_spatial_lora = random.uniform(0, 1) < 0.2 and not mask_temporal_lora
                else:
                    mask_spatial_lora = True

                with accelerator.autocast():
                    loss_spatial, loss_temporal, latents, init_noise = finetune_unet(batch, step, mask_spatial_lora=mask_spatial_lora, mask_temporal_lora=mask_temporal_lora)

                # Gather the losses across all processes for logging (if we use distributed training).
                if not mask_spatial_lora:
                    avg_loss_spatial = accelerator.gather(loss_spatial.repeat(train_batch_size)).mean()
                    train_loss_spatial += avg_loss_spatial.item() / gradient_accumulation_steps

                if not mask_temporal_lora and train_temporal_lora:
                    avg_loss_temporal = accelerator.gather(loss_temporal.repeat(train_batch_size)).mean()
                    train_loss_temporal += avg_loss_temporal.item() / gradient_accumulation_steps

                # Backpropagate
                if not mask_spatial_lora:
                    accelerator.backward(loss_spatial, retain_graph=True)
                    if spatial_lora_num == 1:
                        optimizer_spatial_list[0].step()
                    else:
                        optimizer_spatial_list[step].step()
                    if spatial_lora_num == 1:
                        lr_scheduler_spatial_list[0].step()
                    else:
                        lr_scheduler_spatial_list[step].step()

                if not mask_temporal_lora and train_temporal_lora:
                    accelerator.backward(loss_temporal)
                    optimizer_temporal.step()

                if lr_scheduler_temporal is not None:
                    lr_scheduler_temporal.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss_temporal}, step=global_step)
                train_loss_temporal = 0.0
                if global_step % checkpointing_steps == 0 and global_step > 0:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        image_encoder,
                        vae,
                        output_dir,
                        lora_manager_spatial,
                        lora_manager_temporal,
                        unet_lora_modules,
                        image_encoder_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            unet.eval()
                            image_encoder.eval()
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(seed)
                            unet_and_text_g_c(unet, image_encoder, False, False)
                            loras = extract_lora_child_module(unet, target_replace_module=["BasicTransformerBlock"])
                            for lora_i in loras:
                                lora_i.scale = 0.0

                            if consistency_controller is not None:
                                attention_util.register_attention_control(
                                    unet,
                                    None,
                                    consistency_train_controller,
                                    find_modules={},
                                    consistency_find_modules=consistency_find_modules,
                                    undo=True
                                )

                            pipeline = P2PStableVideoDiffusionPipeline.from_pretrained(
                                pretrained_model_path,
                                image_encoder=image_encoder,
                                vae=vae,
                                unet=unet
                            )
                            if consistency_controller is not None:
                                pipeline.scheduler = P2PEulerDiscreteScheduler.from_config(pipeline.scheduler.config)

#                            # recalculate inversed noise latent
#                            if any([np > 0. for np in validation_data.noise_prior]):
#                                pixel_values_for_inv = batch['pixel_values_for_inv'].to('cuda', dtype=torch.float16)
#                                batch['inversion_noise'] = inverse_video(pipeline, batch['latents_for_inv'], 25, pixel_values_for_inv[:,0,:,:,:])

                            preset_noises = []
                            for noise_prior in validation_data.noise_prior:
                                if noise_prior > 0:
                                    assert batch['inversion_noise'] is not None, "inversion_noise should not be None when noise_prior > 0"
                                    preset_noise = (noise_prior) ** 0.5 * batch['inversion_noise'] + ( 
                                        1-noise_prior) ** 0.5 * torch.randn_like(batch['inversion_noise'])
                                    #print("preset noise", torch.mean(preset_noise), torch.std(preset_noise))
                                else:
                                    preset_noise = None
                                preset_noises.append( preset_noise )

                            for val_img_idx in range(len(validation_images)):
                                for i in range(len(preset_noises)):

                                    if consistency_controller is not None:
                                        consistency_edit_controller = attention_util.ConsistencyAttentionControl(
                                             additional_attention_store=consistency_edit_controller_list[val_img_idx],
                                             use_inversion_attention=False,
                                             save_self_attention=False,
                                             save_latents=False,
                                             disk_store=True
                                        )
                                        attention_util.register_attention_control(
                                            pipeline.unet,
                                            None,
                                            consistency_edit_controller,
                                            find_modules={},
                                            consistency_find_modules=consistency_find_modules,
                                        )
                                        pipeline.scheduler.controller = [consistency_edit_controller] 

                                    preset_noise = preset_noises[i]
                                    save_filename = f"step_{global_step}_noise_{i}_{val_img_idx}"

                                    out_file = f"{output_dir}/samples/{save_filename}.mp4"

                                    val_img = validation_images[val_img_idx]
                                    edited_firstframe = edited_firstframes[val_img_idx]
                                    original_res = val_img.size
                                    resctrl = ResolutionControl(
                                        (original_res[1],original_res[0]),
                                        (validation_data.height, validation_data.width),
                                        validation_data.get("pad_to_fit", False),
                                        fill=0
                                    )
                                    
                                    #val_img = Image.open("white.png").convert("RGB")
                                    val_img = resctrl(val_img)
                                    edited_firstframe = resctrl(edited_firstframe)

                                    with torch.no_grad():
                                        video_frames = pipeline(
                                            val_img,
                                            edited_firstframe=edited_firstframe,
                                            image_latents=validation_images_latents[val_img_idx],
                                            width=validation_data.width,
                                            height=validation_data.height,
                                            num_frames=batch["pixel_values"].shape[1],
                                            decode_chunk_size=8,
                                            motion_bucket_id=127,
                                            fps=validation_data.get('fps', 7),
                                            noise_aug_strength=0.02,
                                            generator=generator,
                                            num_inference_steps=validation_data.num_inference_steps,
                                            latents=preset_noise
                                        ).frames
                                    export_to_video(video_frames, out_file, validation_data.get('fps', 7), resctrl)
                                    if consistency_controller is not None:
                                        attention_util.register_attention_control(
                                            pipeline.unet,
                                            None,
                                            consistency_edit_controller,
                                            find_modules={},
                                            consistency_find_modules=consistency_find_modules,
                                            undo=True
                                        )
                                        consistency_edit_controller.delete()
                                        del consistency_edit_controller
                                    logger.info(f"Saved a new sample to {out_file}")
                            if consistency_controller is not None:
                                attention_util.register_attention_control(
                                    unet,
                                    None,
                                    consistency_train_controller,
                                    find_modules={},
                                    consistency_find_modules=consistency_find_modules,
                                )
                            del pipeline
                            torch.cuda.empty_cache()

                    unet_and_text_g_c(
                        unet,
                        image_encoder,
                        gradient_checkpointing,
                        image_encoder_gradient_checkpointing
                    )

            if loss_temporal is not None:
                accelerator.log({"loss_temporal": loss_temporal.detach().item()}, step=step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            pretrained_model_path,
            global_step,
            accelerator,
            unet,
            image_encoder,
            vae,
            output_dir,
            lora_manager_spatial,
            lora_manager_temporal,
            unet_lora_modules,
            image_encoder_lora_modules,
            is_checkpoint=False,
            save_pretrained_model=save_pretrained_model
        )
    accelerator.end_training()

    if consistency_controller is not None:
        consistency_train_controller.delete()
        del consistency_train_controller


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config_multi_videos.yaml')
    args = parser.parse_args()
    train_motion_lora(**OmegaConf.load(args.config))
