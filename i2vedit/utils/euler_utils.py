import os
import numpy as np
from PIL import Image
from typing import Union
import copy
from scipy.stats import anderson

import torch

from tqdm import tqdm
from diffusers import StableVideoDiffusionPipeline
from i2vedit.prompt_attention import attention_util

# Euler Inversion
@torch.no_grad()
def init_image(image, firstframe, pipeline):
    if isinstance(image, torch.Tensor):
        height, width = image.shape[-2:]
        image = (image + 1) / 2. * 255.
        image = image.type(torch.uint8).squeeze().permute(1,2,0).cpu().numpy()
        image = Image.fromarray(image)
    if isinstance(firstframe, torch.Tensor):
        firstframe = (firstframe + 1) / 2. * 255.
        firstframe = firstframe.type(torch.uint8).squeeze().permute(1,2,0).cpu().numpy()
        firstframe = Image.fromarray(firstframe)

    device = pipeline._execution_device
    image_embeddings = pipeline._encode_image(firstframe, device, 1, False)
    image = pipeline.image_processor.preprocess(image, height=height, width=width)
    firstframe = pipeline.image_processor.preprocess(firstframe, height=height, width=width)
    #print(image.dtype)
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype)
    image = image + 0.02 * noise
    firstframe = firstframe + 0.02 * noise
    #print(image.dtype)
    image_latents = pipeline._encode_vae_image(image, device, 1, False) 
    firstframe_latents = pipeline._encode_vae_image(firstframe, device, 1, False)
    image_latents = image_latents.to(image_embeddings.dtype)
    firstframe_latents = firstframe_latents.to(image_embeddings.dtype)

    return image_embeddings, image_latents, firstframe_latents


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], sigma, sigma_next,
              sample: Union[torch.FloatTensor, np.ndarray], euler_scheduler, controller=None, consistency_controller=None):
    pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    if controller is not None:
        pred_original_sample = controller.step_callback(pred_original_sample)
    if consistency_controller is not None:
        pred_original_sample = consistency_controller.step_callback(pred_original_sample)
    #print("sample", sample.mean(), sample.std(), "pred_original_sample", pred_original_sample.mean()) 
    #pred_original_sample = sample.mean() - pred_original_sample.mean() + pred_original_sample
    next_sample = sample + (sigma_next - sigma) * (sample - pred_original_sample) / sigma
    #print(sigma, sigma_next)
    #print("next sample", torch.mean(next_sample), torch.std(next_sample))
    return next_sample


def get_model_pred_single(latents, t, image_embeddings, added_time_ids, unet):
    noise_pred = unet(
        latents, 
        t, 
        encoder_hidden_states=image_embeddings,
        added_time_ids=added_time_ids,
        return_dict=False,
        )[0]
    return noise_pred

@torch.no_grad()
def euler_loop(pipeline, euler_scheduler, latents, num_inv_steps, image, firstframe, controller=None, consistency_controller=None):
    device = pipeline._execution_device

    # prepare image conditions
    image_embeddings, image_latents, firstframe_latents = init_image(image, firstframe, pipeline)
    skip = 1#latents.shape[1]
    image_latents = torch.cat(
        [
         image_latents.unsqueeze(1).repeat(1, skip, 1, 1, 1),
         firstframe_latents.unsqueeze(1).repeat(1, latents.shape[1]-skip, 1, 1, 1)
        ],
        dim=1
    )
    #image_latents = image_latents.unsqueeze(1).repeat(1, latents.shape[1], 1, 1, 1)

    # Get Added Time IDs
    added_time_ids = pipeline._get_add_time_ids(
        8,
        127,
        0.02,
        image_embeddings.dtype,
        1,
        1,
        False
    )
    added_time_ids = added_time_ids.to(device)

    # Prepare timesteps
    euler_scheduler.set_timesteps(num_inv_steps, device=device)
    sigmas_0 = euler_scheduler.sigmas[-2] * euler_scheduler.sigmas[-2] / euler_scheduler.sigmas[-3]
    timesteps = torch.cat([euler_scheduler.timesteps[1:],torch.Tensor([0.25 * sigmas_0.log()]).to(device)]) 
    sigmas = copy.deepcopy(euler_scheduler.sigmas)
    sigmas[-1] = sigmas_0
    #print(sigmas)

    # prepare latents
    all_latent = [latents]
    latents = latents.clone().detach()

    for i in tqdm(range(num_inv_steps)):
        t = timesteps[len(timesteps) -i -1]
        sigma = sigmas[len(sigmas) -i -1]
        sigma_next = sigmas[len(sigmas) -i -2]
        latent_model_input = latents
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        model_pred = get_model_pred_single(latent_model_input, t, image_embeddings, added_time_ids, pipeline.unet)
        latents = next_step(model_pred, sigma, sigma_next, latents, euler_scheduler, controller=controller, consistency_controller=consistency_controller)
        all_latent.append(latents)
    all_latent[-1] = all_latent[-1] / ((sigmas[0]**2 + 1)**0.5)
    return all_latent


@torch.no_grad()
def euler_inversion(pipeline, euler_scheduler, video_latent, num_inv_steps, image, firstframe, controller=None, consistency_controller=None):
    euler_latents = euler_loop(pipeline, euler_scheduler, video_latent, num_inv_steps, image, firstframe, controller=controller, consistency_controller=consistency_controller)
    return euler_latents

from diffusers import EulerDiscreteScheduler
from .model_utils import tensor_to_vae_latent, load_primary_models, handle_memory_attention

def inverse_video(
    pretrained_model_path, 
    video, 
    keyframe,
    firstframe,
    num_steps, 
    resctrl=None, 
    sard=None, 
    enable_xformers_memory_efficient_attention=True,
    enable_torch_2_attn=False,
    store_controller = None,
    consistency_store_controller = None,
    find_modules={},
    consistency_find_modules={},
    sarp_noise_scale=0.002,
):
    dtype = torch.float32

    # check if inverted latents exists
    for _controller in [store_controller, consistency_store_controller]:
        if _controller is not None:
            if os.path.exists(os.path.join(_controller.store_dir, "inverted_latents.pt")):
                euler_inv_latent = torch.load(os.path.join(_controller.store_dir, "inverted_latents.pt")).to("cuda", dtype)       
                print(f"Successfully load inverted latents from {os.path.join(_controller.store_dir, 'inverted_latents.pt')}")
                return euler_inv_latent
            
    # prepare model, Load scheduler, tokenizer and models.
    noise_scheduler, feature_extractor, image_encoder, vae, unet = load_primary_models(pretrained_model_path)
       
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    vae.to('cuda', dtype=dtype)
    unet.to('cuda')
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet
    )
    pipe.image_encoder.to('cuda')

    attention_util.register_attention_control(
        pipe.unet,
        store_controller,
        consistency_store_controller,
        find_modules=find_modules,
        consistency_find_modules=consistency_find_modules
    )
    if store_controller is not None:
        store_controller.LOW_RESOURCE = True

    video_for_inv = torch.cat([firstframe,keyframe,video],dim=1).to(dtype)
    #print(video_for_inv.shape)
    if resctrl is not None:
        video_for_inv = resctrl(video_for_inv)
    if sard is not None:
        indx = sard.detection(video_for_inv, 0.001)
        #import cv2
        #cv2.imwrite("indx.png", indx[0,0,:,:,:].permute(1,2,0).type(torch.uint8).cpu().numpy()*255)
        noise = torch.randn(video_for_inv.shape, device=video.device, dtype=video.dtype)
        video_for_inv[indx] = video_for_inv[indx] + noise[indx] * sarp_noise_scale
        video_for_inv = video_for_inv.clamp(-1,1)

    firstframe, keyframe, video_for_inv = video_for_inv.tensor_split([1,2],dim=1)

    #print("video for inv", video_for_inv.mean(), video_for_inv.std())
    latents_for_inv = tensor_to_vae_latent(video_for_inv, vae)
    #noise = torch.randn(latents_for_inv.shape, device=video.device, dtype=video.dtype)
    #latents_for_inv = latents_for_inv + noise * sarp_noise_scale
    #print("video latent for inv", latents_for_inv.mean(), latents_for_inv.std(), latents_for_inv.shape)

    euler_inv_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    euler_inv_scheduler.set_timesteps(num_steps)

    euler_inv_latent = euler_inversion(
        pipe, euler_inv_scheduler, video_latent=latents_for_inv.to(pipe.device),
        num_inv_steps=num_steps, image=keyframe[:,0,:,:,:], firstframe=firstframe[:,0,:,:,:], controller=store_controller, consistency_controller=consistency_store_controller)[-1]

    torch.cuda.empty_cache()
    del pipe
    
    #res = anderson(euler_inv_latent.cpu().view(-1).numpy())
    #print(euler_inv_latent.mean(), euler_inv_latent.std())
    #print(res.statistic)
    #print(res.critical_values)
    #print(res.significance_level)

    # save inverted latents
    for _controller in [store_controller, consistency_store_controller]:
        if _controller is not None:
            torch.save(euler_inv_latent, os.path.join(_controller.store_dir, "inverted_latents.pt"))
            break

    return euler_inv_latent
