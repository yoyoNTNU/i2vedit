import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch
from torchvision.transforms import Resize, Pad, InterpolationMode, ToTensor

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

def pad_with_ratio(frames, res, fill=0):
    process = False
    if not isinstance(frames, torch.Tensor):
        frames = ToTensor()(frames).unsqueeze(0)
        process = True
    _, _, ih, iw = frames.shape
#    print("ih, iw", ih, iw)
    i_ratio = ih / iw
    h, w = res
#    print("h,w", h ,w)
    n_ratio = h / w
    if i_ratio > n_ratio:
        nw = int(ih / h * w)
#        print("nw", nw)
        frames = Pad((nw - iw)//2, fill=fill)(frames)
        frames = frames[...,(nw - iw)//2:-(nw - iw)//2,:]
    else:
        nh = int(iw / w * h)
        frames = Pad((nh - ih)//2, fill=fill)(frames)
        frames = frames[...,:,(nh - ih)//2:-(nh - ih)//2]
#    print("after pad", frames.shape)
    if process:
        frames = (frames * 255.).type(torch.uint8).permute(0,2,3,1).squeeze().cpu().numpy()
        frames = Image.fromarray(frames)
    return frames

def return_to_original_res(frames, res, pad_to_fix=False):
    process = False
    if not isinstance(frames, torch.Tensor):
        frames = ToTensor()(frames).unsqueeze(0)
        process = True
    
#    print("original res", res)
    _, _, h, w = frames.shape
#    print("h w", h, w)
    n_ratio = h / w
    ih, iw = res
    i_ratio = ih / iw
    if pad_to_fix:
        if i_ratio > n_ratio:
            nw = int(ih / h * w)
            frames = Resize((ih, iw+2*(nw - iw)//2), interpolation=InterpolationMode.BICUBIC, antialias=True)(frames)
            frames = frames[...,:,(nw - iw)//2:-(nw - iw)//2]
        else:
            nh = int(iw / w * h)
            frames = Resize((ih+2*(nh - ih)//2, iw), interpolation=InterpolationMode.BICUBIC, antialias=True)(frames)

            frames = frames[...,(nh - ih)//2:-(nh - ih)//2,:]
    else:
        frames = Resize((ih, iw), interpolation=InterpolationMode.BICUBIC, antialias=True)(frames)

    if process:
        frames = (frames * 255.).type(torch.uint8).permute(0,2,3,1).squeeze().cpu().numpy()
        frames = Image.fromarray(frames)

    return frames

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids


def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()


def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch, pad_to_fix=False, use_aug=False):
    use_aug = False
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        if not pad_to_fix:
            vr = decord.VideoReader(vid_path, width=w, height=h)
            video = get_frame_batch(vr, use_aug=use_aug)
        else:
            vr = decord.VideoReader(vid_path)
            video = get_frame_batch(vr, use_aug=use_aug)
            video = pad_with_ratio(video, (h, w))
            video = T.transforms.Resize((h, w), antialias=True)(video)

    return video, vr


# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.train_data = self.load_from_json(json_path, json_data)

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = nested_data['clip_path'] if 'clip_path' in nested_data else None
        
        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            self.train_data = []
            print("Non-existant JSON path. Skipping.")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, vr):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.frame_step, 
            self.n_sample_frames
        )
    
    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames

        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = self.sample_start_idx

        return idx

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        # width, height = self.width, self.height
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        frame_range = self.get_frame_range(vr)
        frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def train_data_batch(self, index):

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:

            vid_data = self.train_data[index]

            clip_path = vid_data['clip_path']
            
            # Get video prompt
            prompt = vid_data['prompt']

            video, _ = self.process_video_wrapper(clip_path)

            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids

         # Assign train data
        train_data = self.train_data[index]
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None

        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            width: int = 256,
            height: int = 256,
            inversion_width: int = 256,
            inversion_height: int = 256,
            start_t: float=0,
            end_t: float=-1,
            sample_fps: int=-1,
            single_video_path: str = "",
            refer_image_path: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            pad_to_fix: bool = False,
            use_aug: bool = False,
            **kwargs
    ):
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.start_t = start_t
        self.end_t = end_t
        self.output_fps = sample_fps

        self.single_video_path = single_video_path
        self.refer_image_path = refer_image_path

        self.width = width
        self.height = height
        self.inversion_width = inversion_width
        self.inversion_height = inversion_height

        self.pad_to_fix = pad_to_fix

        self.use_aug = use_aug
        #self.data_augment = ControlNetDataAugmentation() 

    def create_video_chunks(self):
        output_fps = self.output_fps
        start_t = self.start_t
        end_t = self.end_t
        vr = decord.VideoReader(self.single_video_path)
        initial_fps = vr.get_avg_fps()
        if output_fps == -1:
            output_fps = int(initial_fps)
        if end_t == -1:
            end_t = len(vr) / initial_fps
        else:
            end_t = min(len(vr) / initial_fps, end_t)
        assert 0 <= start_t < end_t
        assert output_fps > 0
        start_f_ind = int(start_t * initial_fps)
        end_f_ind = int(end_t * initial_fps)
        num_f = int((end_t - start_t) * output_fps)
        sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
        self.frames = [sample_idx]
        return self.frames

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_frame_batch(self, vr, resize=None, use_aug=False):
        index = self.index
        frames = vr.get_batch(self.frames[self.index])

        if use_aug:
            frames = self.data_augment.augment(frames)
            print(frames.min(), frames.max())

        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch,
                self.pad_to_fix,
                self.use_aug
            )
        video_for_inversion, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.inversion_width, 
                self.inversion_height, 
                self.get_frame_buckets, 
                self.get_frame_batch,
                self.pad_to_fix
            )
        
        return video, video_for_inversion, vr 

    def image_batch(self):
        train_data = self.refer_image_path
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)

        return img

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, video_for_inv, _ = self.process_video_wrapper(train_data)

            return video, video_for_inv
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):

        video, video_for_inv = self.single_video_batch(index)
        image = self.image_batch()
        motion_values = torch.Tensor([127.])

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "pixel_values_for_inv": (video_for_inv / 127.5 - 1.0),
            "refer_pixel_values": (image / 127.5 - 1.0),
            "motion_values": motion_values,
            'dataset': self.__getname__()
        }

        return example


class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=16)

        prompt = get_text_prompt(
            file_path=train_data,
            text_prompt=self.single_img_prompt,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        img, prompt, prompt_ids = self.image_batch(index)
        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        return video, vr
    
    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):

        video, _ = self.process_video_wrapper(self.video_files[index])

        prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video[0] / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt, 'dataset': self.__getname__()}


class CachedDataset(Dataset):
    def __init__(self,cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cuda:0')
        return cached_latent
