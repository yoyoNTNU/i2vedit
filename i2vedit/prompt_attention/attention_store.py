"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import shutil
import torch
import torch.nn.functional as F
from packaging import version
from einops import rearrange
import math

from i2vedit.prompt_attention.common.util import get_time_string

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

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE or 'mask' in place_in_unet:
                # For inversion without null text file 
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # For classifier-free guidance scale!=1
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1

        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def step_callback(self, x_t):

        x_t = super().step_callback(x_t)
        if self.save_latents:
            self.latents_store.append(x_t.cpu().detach())
        return x_t
    
    @staticmethod
    def get_empty_store():
        return {"down_spatial_q_cross": [], "mid_spatial_q_cross": [], "up_spatial_q_cross": [],
                "down_spatial_k_cross": [], "mid_spatial_k_cross": [], "up_spatial_k_cross": [],
                "down_spatial_mask_cross": [], "mid_spatial_mask_cross": [], "up_spatial_mask_cross": [],
                "down_temporal_cross": [], "mid_temporal_cross": [], "up_temporal_cross": [],
                "down_spatial_q_self": [],  "mid_spatial_q_self": [],  "up_spatial_q_self": [],
                "down_spatial_k_self": [],  "mid_spatial_k_self": [],  "up_spatial_k_self": [],
                "down_spatial_mask_self": [],  "mid_spatial_mask_self": [],  "up_spatial_mask_self": [],
                "down_spatial_self": [],  "mid_spatial_self": [],  "up_spatial_self": [],
                "down_temporal_self": [],  "mid_temporal_self": [],  "up_temporal_self": []}

    @staticmethod
    def get_empty_cross_store():
        return {"down_spatial_q_cross": [], "mid_spatial_q_cross": [], "up_spatial_q_cross": [],
                "down_spatial_k_cross": [], "mid_spatial_k_cross": [], "up_spatial_k_cross": [],
                "down_spatial_mask_cross": [], "mid_spatial_mask_cross": [], "up_spatial_mask_cross": [],
                "down_temporal_cross": [], "mid_temporal_cross": [], "up_temporal_cross": [],
                }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= 8*9*8*16:  # avoid memory overload
            # print(f"Store attention map {key} of shape {attn.shape}")
            if (is_cross or self.save_self_attention or 'mask' in key):
                if False:#attn.shape[-2] >= 4*9*4*16:
                    append_tensor = attn.cpu().detach()
                else:
                    append_tensor = attn
                self.step_store[key].append(copy.deepcopy(append_tensor))
                # FIXME: Are these deepcopy all necessary?
                # self.step_store[key].append(append_tensor)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = {key: self.step_store[key] for key in self.step_store if 'mask' in key}
        else:
            for key in self.attention_store:
                if 'mask' in key:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
        
        if self.disk_store:
            path = self.store_dir + f'/{self.cur_step:03d}.pt'
            if self.load_attention_store is None:
                torch.save(copy.deepcopy(self.step_store), path)
                self.attention_store_all_step.append(path)
        else:
            self.attention_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store if 'mask' in key}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        if self.disk_store:
            if self.load_attention_store is not None:
                flist = sorted(os.listdir(self.load_attention_store), key=lambda x: int(x[:-3]))
                self.attention_store_all_step = [
                    os.path.join(self.load_attention_store, fn) for fn in flist
                ]
        self.attention_store = {}

    def __init__(self, 
                 save_self_attention:bool=True, 
                 save_latents:bool=True,
                 disk_store=False,
                 load_attention_store:str=None,
                 store_path:str=None
        ):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if load_attention_store is not None:
            if not os.path.exists(load_attention_store):
                print(f"can not load attentions from {load_attention_store}: file doesn't exist.")
                load_attention_store = None
            else:
                assert self.disk_store, f"can not load attentions from {load_attention_store} because disk_store is disabled."
        self.attention_store_all_step = []
        if self.disk_store:
            if load_attention_store is not None:
                self.store_dir = load_attention_store
                flist = sorted([fpath for fpath in os.listdir(load_attention_store) if "inverted" not in fpath], key=lambda x: int(x[:-3]))
                self.attention_store_all_step = [
                    os.path.join(load_attention_store, fn) for fn in flist
                ]
            else:
                if store_path is None:
                    time_string = get_time_string()
                    path = f'./trash/{self.__class__.__name__}_attention_cache_{time_string}'
                else:
                    path = store_path
                os.makedirs(path, exist_ok=True)
                self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []

        self.save_latents = save_latents
        self.load_attention_store = load_attention_store

    def delete(self):
        if self.disk_store:
            try:
                shutil.rmtree(self.store_dir) 
                print(f"Successfully remove {self.store_dir}")
            except:
                print(f"Fail to remove {self.store_dir}")

    def attention_control(
            self, place_in_unet, attention_type, is_cross, 
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False
        ):
        if attention_type == "temporal":

            return self.temporal_attention_control(
                place_in_unet, attention_type, is_cross,
                q, k, v, attn_mask, dropout_p=0.0, is_causal=False
            )

        elif attention_type == "spatial":
            
            return self.spatial_attention_control(
                place_in_unet, attention_type, is_cross,
                q, k, v, attn_mask, dropout_p=0.0, is_causal=False
            )

    def temporal_attention_control(
            self, place_in_unet, attention_type, is_cross, 
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False
        ):
                
        h = q.shape[1]
        q, k, v = map(lambda t: rearrange(t, "b h n d -> (b h) n d"), (q, k, v))
        attention_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
            q,
            k.transpose(-1, -2),
            beta=0,
            alpha=1 / math.sqrt(q.size(-1)),
        )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attention_scores = attention_scores + attn_mask

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(v.dtype)

        # START OF CORE FUNCTION
        # Record during inversion and edit the attention probs during editing
        attention_probs = rearrange(
            self.__call__(
               rearrange(attention_probs, "(b h) n d -> b h n d", h=h), 
                is_cross, 
                f'{place_in_unet}_{attention_type}'
            ),
            "b h n d -> (b h) n d"
        )
        # END OF CORE FUNCTION
    
        # compute attention output
        hidden_states = torch.bmm(attention_probs, v)

        # reshape hidden_states
        hidden_states = rearrange(hidden_states, "(b h) n d -> b h n d", h=h)

        return hidden_states

    def spatial_attention_control(
            self, place_in_unet, attention_type, is_cross, 
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False
        ):

        q = self.__call__(q, is_cross, f"{place_in_unet}_{attention_type}_q")
        k = self.__call__(k, is_cross, f"{place_in_unet}_{attention_type}_k")

        hidden_states = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

        return hidden_states
