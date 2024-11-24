import os
import decord
import imageio
import numpy as np
import PIL
from PIL import Image
from einops import rearrange, repeat

from torchvision.transforms import Resize, Pad, InterpolationMode, ToTensor, InterpolationMode

import torch
import torch.nn as nn
from torch.utils.data import Dataset

#from i2vedit.utils.augment import ControlNetDataAugmentation, ColorDataAugmentation
# from utils.euler_utils import tensor_to_vae_latent

class ResolutionControl(object):

    def __init__(self, input_res, output_res, pad_to_fit=False, fill=0, **kwargs):
    
        self.ih, self.iw = input_res
        self.output_res = output_res
        self.pad_to_fit = pad_to_fit
        self.fill=fill
        
    def pad_with_ratio(self, frames, res, fill=0):
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            _, _, ih, iw = frames.shape
        elif isinstance(frames, PIL.Image.Image):
            iw, ih = frames.size
        assert ih == self.ih and iw == self.iw, "resolution doesn't match."
        #print("ih, iw", ih, iw)
        i_ratio = ih / iw
        h, w = res
        #print("h,w", h ,w)
        n_ratio = h / w
        if i_ratio > n_ratio:
            nw = int(ih / h * w)
            #print("nw", nw)
            frames = Pad(((nw - iw)//2,0), fill=fill)(frames)
        else:
            nh = int(iw / w * h)
            frames = Pad((0,(nh - ih)//2), fill=fill)(frames)
        #print("after pad", frames.shape)
        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)
        
        return frames

    def return_to_original_res(self, frames):
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            _, _, h, w = frames.shape
        elif isinstance(frames, PIL.Image.Image):
            w, h = frames.size
        #print("original res", (self.ih, self.iw))
        #print("current res", (h, w))
        assert h == self.output_res[0] and w == self.output_res[1], "resolution doesn't match."
        n_ratio = h / w
        ih, iw = self.ih, self.iw
        i_ratio = ih / iw
        if self.pad_to_fit:
            if i_ratio > n_ratio:
                nw = int(ih / h * w)
                frames = Resize((ih, iw+2*(nw - iw)//2), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
                if isinstance(frames, torch.Tensor):
                    frames = frames[...,:,(nw - iw)//2:-(nw - iw)//2]
                elif isinstance(frames, PIL.Image.Image):
                    frames = frames.crop(((nw - iw)//2,0,iw+(nw - iw)//2,ih))              
            else:
                nh = int(iw / w * h)
                frames = Resize((ih+2*(nh - ih)//2, iw), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
                if isinstance(frames, torch.Tensor):
                    frames = frames[...,(nh - ih)//2:-(nh - ih)//2,:]
                elif isinstance(frames, PIL.Image.Image):
                    frames = frames.crop((0,(nh - ih)//2,iw,ih+(nh - ih)//2))
        else:
            frames = Resize((ih, iw), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)

        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)

        return frames

    def __call__(self, frames):
        if self.pad_to_fit:
            frames = self.pad_with_ratio(frames, self.output_res, fill=self.fill)
        
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            frames = (frames + 1) / 2.

        frames = Resize(tuple(self.output_res), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)
            frames = frames * 2 - 1

        return frames

    def callback(self, frames):
        return self.return_to_original_res(frames)

class VideoIO(object):

    def __init__(
        self,
        video_path,
        keyframe_paths,
        output_dir,
        device,
        dtype,
        start_t:int=0, 
        end_t:int=-1, 
        sample_fps:int=-1, 
        chunk_size: int=14, 
        overlay_size: int=-1,
        normalize: bool=True,
        output_fps: int=-1,
        save_sampled_video: bool=True,
        **kwargs
    ):
        self.video_path = video_path
        self.keyframe_paths = keyframe_paths
        self.device = device
        self.dtype = dtype
        self.start_t = start_t
        self.end_t = end_t
        self.sample_fps = sample_fps
        self.chunk_size = chunk_size
        self.overlay_size = overlay_size
        self.normalize = normalize
        self.save_sampled_video = save_sampled_video

        

        vr = decord.VideoReader(video_path)
        initial_fps = vr.get_avg_fps()
        self.initial_fps = initial_fps

        if output_fps == -1: output_fps = initial_fps

        self.video_writer_list = []
        for keyframe_path in keyframe_paths:
            fname, ext = os.path.splitext(os.path.basename(keyframe_path))
            output_video_path = os.path.join(output_dir, fname+".mp4")
            self.video_writer_list.append( imageio.get_writer(output_video_path, fps=output_fps) )
        
        if save_sampled_video:
            fname, ext = os.path.splitext(os.path.basename(video_path))
            output_sampled_video_path = os.path.join(output_dir, fname+f"_from{start_t}s_to{end_t}s{ext}")
            self.sampled_video_writer = imageio.get_writer(output_sampled_video_path, fps=output_fps)

    def read_keyframe_iter(self):
        for keyframe_path in self.keyframe_paths:
            image = Image.open(keyframe_path).convert("RGB") 
            yield image

    def read_video_iter(self):
        vr = decord.VideoReader(self.video_path)
        if self.sample_fps == -1: self.sample_fps = self.initial_fps
        if self.end_t == -1: 
            self.end_t = len(vr) / self.initial_fps
        else:
            self.end_t = min(len(vr) / self.initial_fps, self.end_t)
        if self.overlay_size == -1: self.overlay_size = 0
        assert 0 <= self.start_t < self.end_t
        assert self.sample_fps > 0

        start_f_ind = int(self.start_t * self.initial_fps)
        end_f_ind = int(self.end_t * self.initial_fps)
        num_f = int((self.end_t - self.start_t) * self.sample_fps)
        sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
        print("sample_idx", sample_idx)

        assert len(sample_idx) > 0, f"sample_idx is empty!"

        begin_frame_idx = 0
        while begin_frame_idx < len(sample_idx):
            self.begin_frame_idx = begin_frame_idx
            begin_frame_idx = max(begin_frame_idx - self.overlay_size, 0)
            next_frame_idx = min(begin_frame_idx + self.chunk_size, len(sample_idx))

            video = vr.get_batch(sample_idx[begin_frame_idx:next_frame_idx])
            begin_frame_idx = next_frame_idx

            if self.save_sampled_video:
                overlay_size = 0 if self.begin_frame_idx == 0 else self.overlay_size 
                for frame in video[overlay_size:]:
                    self.sampled_video_writer.append_data(frame.detach().cpu().numpy())

            video = torch.Tensor(video).to(self.device).to(self.dtype)
            video = rearrange(video, "f h w c -> f c h w")

            if self.normalize:
                video = video / 127.5 - 1.0
            
            yield video

    def write_video(self, video, video_id, resctrl: ResolutionControl = None):
        '''
        video: 
        '''
        overlay_size = 0 if self.begin_frame_idx == 0 else self.overlay_size 
        for img in video[overlay_size:]:
            if resctrl is not None:
                img = resctrl.callback(img)
            self.video_writer_list[video_id].append_data(np.array(img))

    def close(self):
        for video_writer in self.video_writer_list:
            video_writer.close()
        if self.save_sampled_video:
            self.sampled_video_writer.close()
        self.begin_frame_idx = 0


class SingleClipDataset(Dataset):

#    data_aug_class = {
#        "rsfnet": ColorDataAugmentation,
#        "controlnet": ControlNetDataAugmentation
#    }

    def __init__(
        self,
        inversion_noise,
        video_clip,
        keyframe,
        firstframe,
        height,
        width,
        use_data_aug=None,
        pad_to_fit=False,
        keyframe_latent=None
    ):
        
        self.resctrl = ResolutionControl(video_clip.shape[-2:],(height,width),pad_to_fit,fill=-1)

        video_clip = rearrange(video_clip, "1 f c h w -> f c h w")
        keyframe = rearrange(keyframe, "1 f c h w -> f c h w")
        firstframe = rearrange(firstframe, "1 f c h w -> f c h w")
        
        if inversion_noise is not None:
            inversion_noise = rearrange(inversion_noise, "1 f c h w -> f c h w")
        
        if use_data_aug is not None:
            if use_data_aug in self.data_aug_class:
                self.data_augment = self.data_aug_class[use_data_aug]()
                use_data_aug = True
                print(f"Augmentation mode: {use_data_aug} is implemented.")
            else:
                raise NotImplementedError(f"Augmentation mode: {use_data_aug} is not implemented!")
        else:
            use_data_aug = False

        self.video_clip = video_clip
        self.keyframe = keyframe
        self.firstframe = firstframe
        self.inversion_noise = inversion_noise
        self.use_data_aug = use_data_aug
        self.keyframe_latent = keyframe_latent
    
    @staticmethod
    def __getname__(): return 'single_clip'

    def __len__(self):
        return 1

    def __getitem__(self, index):

        motion_values = torch.Tensor([127.])

        pixel_values = self.resctrl(self.video_clip)
        refer_pixel_values = self.resctrl(self.keyframe)
        cross_pixel_values = self.resctrl(self.firstframe)

        if self.use_data_aug:
            print("pixel_values before augment", refer_pixel_values.min(), refer_pixel_values.max())
            #pixel_values, refer_pixel_values, cross_pixel_values = \
            #self.data_augment.augment(
            #    torch.cat([pixel_values, refer_pixel_values, cross_pixel_values], dim=0)
            #).tensor_split([pixel_values.shape[0],pixel_values.shape[0]+refer_pixel_values.shape[0]],dim=0)
            refer_pixel_values = self.data_augment.augment(refer_pixel_values)
            print("pixel_values after augment", refer_pixel_values.min(), refer_pixel_values.max())
            
        outputs = {
            "pixel_values": pixel_values,
            "refer_pixel_values": refer_pixel_values,
            "cross_pixel_values": cross_pixel_values,
            "motion_values": motion_values,
            'dataset': self.__getname__(),
        }

        if self.inversion_noise is not None:
            outputs.update({
                "inversion_noise": self.inversion_noise
            })
        if self.keyframe_latent is not None:
            outputs.update({
                "refer_latents": self.keyframe_latent 
            })
        return outputs
