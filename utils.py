#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import random
# 3rd part packages
from decord import VideoReader
import cv2
import numpy as np
from PIL import Image
from decord.ndarray import array
import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

# local source


def get_new_shape(image, target_w, target_h):
    h, w = image.shape[:2]
    if w / h > target_w / target_h:
        oh = target_h
        ow = int(target_h * w / h)
    else:
        ow = target_w
        oh = int(target_w * h / w)
    return oh, ow


def image_to_video(image, kps, frame_num=30, window_w=512, window_h=768, w_add=20, h_add=20, fix_ratio=0.1):
    target_w = window_w + w_add
    target_h = window_h + h_add

    new_height, new_width = get_new_shape(image, target_w, target_h)

    # 对图片进行resize
    image_resize = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    kps = cv2.resize(kps, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    w_window = new_width - window_w
    h_window = new_height - window_h

    start_w = random.randint(0, w_window)
    start_h = random.randint(0, h_window)
    end_w = random.randint(0, w_window)
    end_h = random.randint(0, h_window)

    w_list = np.linspace(start_w, end_w, frame_num).astype(int)
    h_list = np.linspace(start_h, end_h, frame_num).astype(int)

    if random.random() < fix_ratio:
        w_list = w_list * 0
        h_list = h_list * 0

    frames = []
    kpss = []
    for i in range(frame_num):
        h_start = h_list[i]
        w_start = w_list[i]
        crop = image_resize[h_start:h_start + window_h, w_start:w_start + window_w]
        kps_crop = kps[h_start:h_start + window_h, w_start:w_start + window_w]
        crop = array(crop)
        kps_crop = array(kps_crop)
        frames.append(crop)
        kpss.append(kps_crop)

    return frames, kpss, w_list, h_list, image_resize


def get_video_reader(video_path, kps_path, img_expand_ratio):
    if kps_path.endswith('.mp4'):
        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        # use frame as video
        rnd = random.random()
        if rnd < img_expand_ratio:
            video_length = len(video_reader)
            img_idx = random.randint(0, video_length - 1)
            image = video_reader[img_idx].asnumpy()
            kps = kps_reader[img_idx].asnumpy()
            frames, kpss, _, _, _ = image_to_video(image, kps, fix_ratio=1)
            video_reader = frames
            kps_reader = kpss
    else:
        image = Image.open(video_path).convert('RGB')
        kps = Image.open(kps_path).convert('RGB')

        image = np.array(image)
        kps = np.array(kps)
        # if kps is None:
        #     print(kps_path)
        # if image is None:
        #     print(video_path)

        frames, kpss, _, _, _ = image_to_video(image, kps)

        video_reader = frames
        kps_reader = kpss

    return video_reader, kps_reader




def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def main():
    print('main')


if __name__ == '__main__':
    main()
