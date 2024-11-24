"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import logging
from einops import rearrange, repeat
import math
from inspect import isfunction
from typing import Any, Optional
from packaging import version

from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.utils import USE_PEFT_BACKEND

class AttnControllerProcessor:

    def __init__(self, consistency_controller, controller, place_in_unet, attention_type):
        
        self.consistency_controller = consistency_controller
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.attention_type = attention_type

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.consistency_controller is not None:
            key = self.consistency_controller(
                key, is_cross, f"{self.place_in_unet}_{self.attention_type}_k"
            )
            value = self.consistency_controller(
                value, is_cross, f"{self.place_in_unet}_{self.attention_type}_v"
            )

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.controller is not None:
            hidden_states = self.controller.attention_control(
                self.place_in_unet, self.attention_type, is_cross, 
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_attention_control(
    model, 
    controller=None, 
    consistency_controller=None,
    find_modules = {},
    consistency_find_modules = {},
    undo=False
):
    "Connect a model with a controller"
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    #if controller is None:
    #    controller = DummyController()

    f_keys = list(set(find_modules.keys()).difference(set(consistency_find_modules.keys())))
    c_keys = list(set(consistency_find_modules.keys()).difference(set(find_modules.keys())))
    common_keys = list(set(find_modules.keys()).intersection(set(consistency_find_modules.keys())))
    new_find_modules = {}
    for f_key in f_keys:
        new_find_modules.update({
            f_key: find_modules[f_key]
        })
    new_consistency_find_modules = {}
    for c_key in c_keys:
        new_consistency_find_modules.update({
            c_key: consistency_find_modules[c_key]
        })
    common_modules = {}
    for key in common_keys:
        find_modules[key] = [] if find_modules[key] is None else find_modules[key]
        consistency_find_modules[key] = [] if consistency_find_modules[key] is None else consistency_find_modules[key]
        f_list = list(set(find_modules[key]).difference(set(consistency_find_modules[key])))
        c_list = list(set(consistency_find_modules[key]).difference(set(find_modules[key])))
        common_list = list(set(find_modules[key]).intersection(set(consistency_find_modules[key])))
        if len(f_list) > 0:
            new_find_modules.update({key: f_list})
        if len(c_list) > 0:
            new_consistency_find_modules.update({key: c_list})
        if len(common_list) > 0:
            common_modules.update({key: common_list})

    find_modules = new_find_modules
    consistency_find_modules = new_consistency_find_modules

    print("common_modules", common_modules)
    print("find_modules", find_modules)
    print("consistency_find_modules", consistency_find_modules)
    print("controller", controller, "consistency_controller", consistency_controller)

    def register_recr(net_, count1, count2, place_in_unet):

        if net_[1].__class__.__name__ == 'BasicTransformerBlock':
            attention_type = 'spatial'
        elif net_[1].__class__.__name__ == 'TemporalBasicTransformerBlock':
            attention_type = 'temporal'

        control1, control2 = None, None
        if net_[1].__class__.__name__ in common_modules.keys():
            control1, control2 = consistency_controller, controller
            module_list = common_modules[net_[1].__class__.__name__]
        elif net_[1].__class__.__name__ in find_modules.keys():
            control1, control2 = None, controller
            module_list = find_modules[net_[1].__class__.__name__]
        elif net_[1].__class__.__name__ in consistency_find_modules.keys():
            control1, control2 = consistency_controller, None
            module_list = consistency_find_modules[net_[1].__class__.__name__]

        if any([control is not None for control in [control1, control2]]):

            if module_list is not None and 'attn1' in module_list:
                if undo:
                    net_[1].attn1.set_processor(AttnProcessor2_0())
                else:
                    net_[1].attn1.set_processor(AttnControllerProcessor(control1, control2, place_in_unet, attention_type = attention_type))
                if control1 is not None: count1 += 1
                if control2 is not None: count2 += 1

            if module_list is not None and 'attn2' in module_list:
                if undo:
                    net_[1].attn2.set_processor(AttnProcessor2_0())
                else:
                    net_[1].attn2.set_processor(AttnControllerProcessor(control1, control2, place_in_unet, attention_type = attention_type))
                if control1 is not None: count1 += 1
                if control2 is not None: count2 += 1

            return count1, count2

        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                count1, count2 = register_recr(net, count1, count2, place_in_unet)

        return count1, count2

    cross_att_count1 = 0
    cross_att_count2 = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            c1, c2 = register_recr(net, 0, 0, "down")
            cross_att_count1 += c1
            cross_att_count2 += c2
        elif "up" in net[0]:
            c1, c2 = register_recr(net, 0, 0, "up")
            cross_att_count1 += c1
            cross_att_count2 += c2
        elif "mid" in net[0]:
            c1, c2 = register_recr(net, 0, 0, "mid")
            cross_att_count1 += c1
            cross_att_count2 += c2
    if undo:
        print(f"Number of attention layer unregistered for controller: {cross_att_count2}")
        print(f"Number of attention layer unregistered for consistency_controller: {cross_att_count1}")
    else:
        print(f"Number of attention layer registered for controller: {cross_att_count2}")
        if controller is not None:
            controller.num_att_layers = cross_att_count2
        print(f"Number of attention layer registered for consistency_controller: {cross_att_count1}")
        if consistency_controller is not None:
            consistency_controller.num_att_layers = cross_att_count1
            

