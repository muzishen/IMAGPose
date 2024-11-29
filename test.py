import argparse
from datetime import datetime
from pathlib import Path
import torch
from diffusers import AutoencoderKL, DDIMScheduler

from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from models.unet_3d import UNet3DConditionModel
from models.pose_guider import PoseGuider
from train_stage1_vae_clip import PatchEmbedding

from pipeline.pipeline_pose2img import Pose2ImagePipeline

from dwpose import DWposeDetector
from einops import rearrange, repeat
import numpy as np
import random
from transformers import Dinov2Model
import os

def big2small_image(big_img): # b, h,w, c
    big_img = rearrange(big_img, "b h w c -> b c h w")

    bs, _, height, width = big_img.shape
    image1 = big_img[:, :, :height//2 , :width//2]
    image2 = big_img[:, :, :height//2, width//2:]
    image3 = big_img[:, :, height//2:, :width//2]
    image4 = big_img[:, :, height//2:, width//2:]

    batch_image = torch.stack([image1, image2, image3, image4], dim=0) # f, b, c, h, w
    batch_image = rearrange(batch_image, "f b c h w -> b c f h w")
    return batch_image



def concat_big_img(img_list,  width, height,):
    scale_transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
            ]
        )
    img1, img2, img3, img4 = scale_transform(img_list[0]), scale_transform(img_list[1]), scale_transform(img_list[2]), scale_transform(img_list[3])

    width, height = img1.size

    if  len(img1.getbands()) == 1:
        final_image = Image.new('L', (width * 2, height * 2))
    else:
        final_image = Image.new('RGB', (width * 2, height * 2))


    final_image.paste(img1, (0, 0))
    final_image.paste(img2, (width, 0))
    final_image.paste(img3, (0, height))
    final_image.paste(img4, (width, height))
    return final_image


def tensor2list(bs):
    reshaped_tensor = bs.reshape(-1, bs.size(-1))

    splitted_tensors = torch.split(reshaped_tensor, bs.size(1) * bs.size(2), dim=0)

    x_list = [t.view(bs.size(1), bs.size(2), bs.size(-1)) for t in splitted_tensors]
    return x_list


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def inference_pose(img_path, detector):
    print(img_path)
    pil_image = Image.open(img_path).convert("RGB")
    dwpose_image, score, _, _ = detector(pil_image, output_type='np')
    pil_dwpose_image = Image.fromarray(dwpose_image)

    return pil_dwpose_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/prompts/test_end2end.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--base_root", type=str,default='./data/test_demo')
    parser.add_argument("--person_path", type=str,default='./data/test_demo/imgs/demo10_1.png')
    parser.add_argument("--person_pose_path", type=str,default='./data/test_demo/dwpose/demo10_2_pose.png')
    parser.add_argument("--save_dir_name", type=str,default='./all_logs/test_results')
    parser.add_argument("--denoising_unet_path", type=str, default='./all_logs/stage3_end2end/stage3_end2end_log/denoising_unet-50001.pth')
    parser.add_argument("--pose_guider_path", type=str, default='./all_logs/stage3_end2end/stage3_end2end_log/pose_guider-50001.pth')
    parser.add_argument("--patch_path", type=str, default='./all_logs/stage3_end2end/stage3_end2end_log/patch-50001.pth')
    parser.add_argument("--motion_module_path", type=str, default='./all_logs/stage3_end2end/stage3_end2end_log/motion_module-50001.pth')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if args.denoising_unet_path is not None:
        config.denoising_unet_path = args.denoising_unet_path
    if args.motion_module_path is not None:
        config.motion_module_path = args.motion_module_path
    if args.patch_path is not None:
        config.patch_path = args.patch_path
    if args.pose_guider_path is not None:
        config.pose_guider_path = args.pose_guider_path

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)


    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    detector = DWposeDetector().to("cuda")


    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    patch_encoder = PatchEmbedding(patch_size=16, in_chans=4, embed_dim=768).to(
        dtype=weight_dtype, device="cuda"
    )
    image_enc = Dinov2Model.from_pretrained(
        config.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )

    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    patch_encoder.load_state_dict(
        torch.load(config.patch_path, map_location="cpu"),
    )
    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        patch_encoder=patch_encoder,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    base_root = args.base_root
    save_dir_name = args.save_dir_name
    save_dir = Path(f"{base_root}/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)


    # read frames and kps
    person_paths = [args.person_path,]
    if len(person_paths) == 1:
        person_paths = person_paths * 3
    elif len(person_paths) == 2:

        chosen_element = random.choice(person_paths)

        person_paths.append(chosen_element)
    elif len(person_paths) == 3:
        person_paths = person_paths
    else:
        raise ValueError("The list at least have 1 image")

    person_poses = [args.person_pose_path,]

    if len(person_paths) + len(person_poses) != 4:
        raise ValueError("The length of the list is not 4.")
    else:
        print("The length of the list is 4.")

    black_img_pil = Image.new("RGB", (args.W, args.H), (0, 0, 0))
    if isinstance(person_paths, list):
        ref_img_pils = [Image.open(person_path).convert("RGB") for person_path in person_paths]
        image_mask_pil_image_list = ref_img_pils.copy()
        while len(image_mask_pil_image_list) < 4:
            image_mask_pil_image_list.append(black_img_pil)
    else:
        raise ValueError("Input should be a list of paths (strings).")


    image_mask_big_image = concat_big_img(image_mask_pil_image_list, args.W, args.H)
    pixel_values_image_mask = transforms.ToTensor()(image_mask_big_image)


    pose_pil_image_list = [inference_pose(person_path, detector) for person_path in person_paths]
    pose_pil_image_list += [Image.open(person_pose).convert("RGB") for person_pose in person_poses]
    pose_pil_big_image = concat_big_img(pose_pil_image_list, args.W, args.H)

    pixel_values_pose = transforms.ToTensor()(pose_pil_big_image)

    # setting flag label
    white1 = Image.new("L", (args.W //8, args.H//8), 255)
    black0 = Image.new("L", (args.W//8, args.H//8), 0)
    flag_label =[white1 for _ in ref_img_pils]
    while len(flag_label) < 4:
        flag_label.append(black0)
    flag_label_mask_big_image = concat_big_img(flag_label, args.W//8, args.H//8)

    pixel_values_flag_label = transforms.ToTensor()(flag_label_mask_big_image)

    ref_img_pil = ref_img_pils[0]
    image = pipe(
        ref_image = ref_img_pil,
        pose_image =pixel_values_pose,
        image_mask = pixel_values_image_mask,
        flag_label = pixel_values_flag_label,
        width = width*2,
        height = height*2,
        num_inference_steps = args.steps,
        guidance_scale = args.cfg,
        generator=generator,
    ).images  # b,h,w,c


    grid = image_grid(image, 1, 1)
    grid.save(f"{save_dir}/{person_poses[0].split('/')[-1]}{args.H}x{args.W}_{int(args.cfg)}_{time_str}.png")

if __name__ == "__main__":
    main()