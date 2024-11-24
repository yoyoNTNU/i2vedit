from typing import List
import os
import datetime
import numpy as np
from PIL import Image
from einops import rearrange, repeat
import math

import torch
import torch.nn.functional as F
from packaging import version

from i2vedit.prompt_attention import ptp_utils
from i2vedit.prompt_attention.common.image_util import save_gif_mp4_folder_type
from i2vedit.prompt_attention.attention_store import AttentionStore

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.dim() == 3:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
            elif item.dim() == 4:
                t, h, res_sq, token = item.shape
                if item.shape[2] == num_pixels:
                    cross_maps = item.reshape(len(prompts), t, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
                    
    out = torch.cat(out, dim=-4)
    out = out.sum(-4) / out.shape[-4]
    return out.cpu()


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) 
    decoder = tokenizer.decode
    
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    os.makedirs('trash', exist_ok=True)
    attention_list = []
    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    for j in range(attention_maps.shape[0]):
        images = []
        for i in range(len(tokens)):
            image = attention_maps[j, :, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)
        atten_j = np.concatenate(images, axis=1)
        attention_list.append(atten_j)
    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        video_save_path = f'{save_path}/{now}.gif'
        save_gif_mp4_folder_type(attention_list, video_save_path)
    return attention_list
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def show_avg_difference_maps(
    attention_store: AttentionStore, 
    save_path = None
):
    avg_attention = attention_store.get_average_attention()
    masks = []
    for key in avg_attention:
        if 'mask' in key:
            for cur_pos in range(len(avg_attention[key])):
                mask = avg_attention[key][cur_pos] 
                res = mask.shape[0] / 9
                file_path = os.path.join(
                    save_path, 
                    f"avg_key_{key}_curpos_{cur_pos}_res_{res}_mask.png"
                )
                print(key, cur_pos, mask.shape)
                image = 255 * mask #/ attn.max()
                image = image.cpu().numpy().astype(np.uint8)
                image = Image.fromarray(image) 
                image.save(file_path)
                

    

def show_self_attention(
    attention_store: AttentionStore, 
    steps: List[int], 
    save_path = None,
    inversed = False):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
    """
    #os.system(f"rm -rf {save_path}")
    os.makedirs(save_path, exist_ok=True)
    for step in steps:
        step_in_store = len(attention_store.attention_store_all_step) - step - 1 if inversed else step
        print("step_in_store", step_in_store)
        step_in_store_atten_dict = attention_store.attention_store_all_step[step_in_store]
        if isinstance(step_in_store_atten_dict, str):
            step_in_store_atten_dict = torch.load(step_in_store_atten_dict)

        step_in_store_atten_dict_reorg = {}

        for key in step_in_store_atten_dict:
            if '_q_' not in key and ('_k_' not in key and '_v_' not in key):
                step_in_store_atten_dict_reorg[key] = step_in_store_atten_dict[key]
            elif '_q_' in key:
                step_in_store_atten_dict_reorg[key.replace("_q_","_qxk_")] = \
                [[step_in_store_atten_dict[key][i], \
                  step_in_store_atten_dict[key.replace("_q_","_k_")][i] \
                 ] \
                 for i in range(len(step_in_store_atten_dict[key]))]

        for key in step_in_store_atten_dict_reorg:
            if '_mask_' not in key and '_qxk_' not in key:
                for cur_pos in range(len(step_in_store_atten_dict_reorg[key])):
                    attn = step_in_store_atten_dict_reorg[key][cur_pos]
                    attn = torch.mean(attn, dim=1)
                    s, t, d = attn.shape
                    res = int(np.sqrt(s / (9*16)))
                    attn = attn.reshape(res*9,res*16,t,d).permute(2,0,3,1).reshape(t*res*9,d*res*16)
                    file_path = os.path.join(
                        save_path, 
                        f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}.png"
                    )
                    print(step, key, cur_pos, attn.shape)
                    image = 255 * attn #/ attn.max()
                    image = image.cpu().numpy().astype(np.uint8)
                    image = Image.fromarray(image) 
                    image.save(file_path)

            elif '_mask_' in key:
                for cur_pos in range(len(step_in_store_atten_dict_reorg[key])):
                    mask = step_in_store_atten_dict_reorg[key][cur_pos]
                    res = mask.shape[0] / 9
                    file_path = os.path.join(
                        save_path, 
                        f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}_mask.png"
                    )
                    print(step, key, cur_pos, mask.shape)
                    image = 255 * mask #/ attn.max()
                    image = image.cpu().numpy().astype(np.uint8)
                    image = Image.fromarray(image) 
                    image.save(file_path)

            else:
                for cur_pos in range(len(step_in_store_atten_dict_reorg[key])):
                    q, k = step_in_store_atten_dict_reorg[key][cur_pos] 
                    q = q.to("cuda").type(torch.float32)
                    k = k.to("cuda").type(torch.float32) 
                    res = int(np.sqrt(q.shape[-2] / (9*16)))
                    h = q.shape[1]
                    bs = 1
                    N = q.shape[0] // bs
                    vectors = []
                    vectors_diff = []
                    for i in range(N):
                        attn_prob = calculate_attention_probs(q[i*bs:(i+1)*bs], k[i*bs:(i+1)*bs])
                        print("attn_prob 1", attn_prob.min(), attn_prob.max())
                        attn_prob = torch.mean(attn_prob, dim=2).reshape(h, res*9, res*16)
                        print("attn_prob 2", attn_prob.min(), attn_prob.max())
                        attn_prob = torch.mean(attn_prob, dim=0)
                        print("attn_prob 3", attn_prob.min(), attn_prob.max())
                        vectors.append( attn_prob )
                    for i in range(1, len(vectors)):
                        vectors_diff.append(vectors[i] - vectors[i-1])
                    vectors = torch.cat(vectors, dim=1)
                    vectors_diff = torch.cat(vectors_diff, dim=1)
                    file_path = os.path.join(
                        save_path, 
                        f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}_vector.png"
                    )
                    print(step, key, cur_pos, vectors.shape)
                    image = 255 * vectors / vectors.max()
                    image = image.clamp(0,255).cpu().numpy().astype(np.uint8)
                    image = Image.fromarray(image) 
                    image.save(file_path)

                    file_path = os.path.join(
                        save_path, 
                        f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}_diff.png"
                    )
                    print(step, key, cur_pos, vectors_diff.shape)
                    image = 255 * vectors_diff / vectors_diff.max()
                    image = image.clamp(0,255).cpu().numpy().astype(np.uint8)
                    image = Image.fromarray(image) 
                    image.save(file_path)


#            else:
#                #  只看最后两帧
#                for cur_pos in range(len(step_in_store_atten_dict_reorg[key])):
#                    q, k, v = step_in_store_atten_dict_reorg[key][cur_pos]
#                    q = q[-2:,...].to("cuda")
#                    k = k[-2:,...].to("cuda")
#                    v = v[-2:,...].to("cuda")
#                    res = int(np.sqrt(q.shape[-2] / (9*16)))
#                    attn = calculate_attention_probs(q,k,v)
#                    attn_d = torch.sum(torch.mean(torch.abs(attn[0,...] - attn[1,...]), dim=0), dim=1).reshape(res*9,res*16)
#                    print(step, key, cur_pos, attn_d.shape, attn_d.min(), attn_d.max()) 
#                    file_path = os.path.join(
#                        save_path, 
#                        f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}_attn_d.png"
#                    )
#                    image = (255 * attn_d + 1e-3) / 2.#attn_d.max()
#                    image = image.clamp(0,255).cpu().numpy().astype(np.uint8)
#                    image = Image.fromarray(image) 
#                    image.save(file_path)

def show_self_attention_distance(
    attention_store: List[AttentionStore], 
    steps: List[int], 
    save_path = None,
):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
    """
    os.system(f"rm -rf {save_path}")
    os.makedirs(save_path, exist_ok=True)
    assert len(attention_store) == 2
    for step in steps:
        step_in_store = [len(attention_store[0].attention_store_all_step) - step - 1, step]
        step_in_store_atten_dict = [attention_store[i].attention_store_all_step[step_in_store[i]] \
                                    for i in range(2)]
        step_in_store_atten_dict = [ \
            torch.load(step_in_store_atten_dict[i]) \
            if isinstance(step_in_store_atten_dict[i], str) \
            else step_in_store_atten_dict[i] \
            for i in range(2)]

        step_in_store_atten_dict_reorg = [{},{}]

        for i in range(2):
            item = step_in_store_atten_dict[i]
            for key in item:
                if '_q_' in key:
                    step_in_store_atten_dict_reorg[i][key.replace("_q_","_qxk_")] = \
                    [[step_in_store_atten_dict[i][key][j], \
                      step_in_store_atten_dict[i][key.replace("_q_","_k_")][j] \
                     ] \
                     for j in range(len(step_in_store_atten_dict[i][key]))]

        for key in step_in_store_atten_dict_reorg[1]:
            for cur_pos in range(len(step_in_store_atten_dict_reorg[1][key])):
                q1, k1 = step_in_store_atten_dict_reorg[1][key][cur_pos]
                q0, k0 = step_in_store_atten_dict_reorg[0][key][cur_pos]
                res = int(np.sqrt(q1.shape[-2] / (9*16))) 

                attn_d = calculate_attention_mask(q0, k0, q1, k1, bs=1, device="cuda")
                attn_d = rearrange(attn_d, "b h w -> h (b w)")

                print(step, key, cur_pos, attn_d.shape, "attnd", attn_d.min(), attn_d.max()) 
                file_path = os.path.join(
                    save_path, 
                    f"step_{step}_key_{key}_curpos_{cur_pos}_res_{res}_attn_d.png"
                )
                image = 255 * attn_d#attn_d.max()
                image = image.clamp(0,255).cpu().numpy().astype(np.uint8)
                image = Image.fromarray(image) 
                image.save(file_path)

def calculate_attention_mask(q0, k0, q1, k1, bs=1, device="cuda"):
    q1 = q1.to(device)
    k1 = k1.to(device)
    q0 = q0.to(device)
    k0 = k0.to(device)
    res = int(np.sqrt(q1.shape[-2] / (9*16))) 
    N = q0.shape[0] // bs
    attn_d = []
    for i in range(N):
        attn0 = calculate_attention_probs(q0[bs*i:bs*(i+1),...],k0[bs*i:bs*(i+1),...])
        attn1 = calculate_attention_probs(q1[bs*i:bs*(i+1),...],k1[bs*i:bs*(i+1),...])
        attn_d_i = torch.sum(torch.mean(torch.abs(attn0 - attn1), dim=1), dim=2).reshape(bs,res*9,res*16)
        attn_d.append( attn_d_i )
    attn_d = torch.cat(attn_d, dim=0) / 2.0
    return attn_d.clamp(0,1)

def calculate_attention_probs(q, k, attn_mask=None):
    with sdp_kernel(**BACKEND_MAP[None]):
        h = q.shape[1]
        q, k = map(lambda t: rearrange(t, "b h n d -> (b h) n d"), (q, k))
        
        with torch.autocast("cuda", enabled=False):
            attention_scores = torch.baddbmm(
                torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
                q,
                k.transpose(-1, -2),
                beta=0,
                alpha=1 / math.sqrt(q.size(-1)),
            )
        #print("attention_scores", attention_scores.min(), attention_scores.max())

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attention_scores = attention_scores + attn_mask

        attention_probs = attention_scores.softmax(dim=-1)
        #print("attention_softmax", attention_probs.min(), attention_probs.max())

        # cast back to the original dtype
        attention_probs = attention_probs.to(q.dtype)

        # reshape hidden_states
        attention_probs = rearrange(attention_probs, "(b h) n d -> b h n d", h=h)

#        v = torch.eye(q.shape[-2], device=q.device)
#        v = repeat(v, "... -> b h ...", b=q.shape[0], h=q.shape[1])
#        attention_probs = F.scaled_dot_product_attention(
#            q, k, v, attn_mask=attn_mask
#        )  # scale is dim_head ** -0.5 per default

    return attention_probs
