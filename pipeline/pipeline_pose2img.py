import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor



@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class Pose2ImagePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        patch_encoder,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            patch_encoder=patch_encoder,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.image_mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        pose_image,
        image_mask,
        flag_label,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        image_prompt_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).last_hidden_state
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )



        # num_channels_latents = self.denoising_unet.in_channels
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            image_prompt_embeds.dtype,
            device,
            generator,
        )
        latents = latents.unsqueeze(2) # (bs, 4, f, h', w)
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents to patch embeds
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        patch_ref_latents = self.patch_encoder(ref_image_latents)
        uncond_patch_ref_latents = torch.zeros_like(patch_ref_latents)

        if do_classifier_free_guidance:
            patch_ref_latents = torch.cat(
                [uncond_patch_ref_latents, patch_ref_latents], dim=0
            )

        # Prepare  image mask latents
        image_mask_tensor = self.image_mask_processor.preprocess(
            image_mask, height=height, width=width
        )  # (bs, c, width, height)
        image_mask_tensor = image_mask_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        image_mask_latents = self.vae.encode(image_mask_tensor).latent_dist.mean.unsqueeze(2)
        image_mask_latents = image_mask_latents * 0.18215  # (b, 4, h, w)
        clip_vae_embeds = torch.cat([image_prompt_embeds, patch_ref_latents], dim=1)  # bs, (257+96), 768

        # Prepare pose condition image
        pose_cond_tensor = self.cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        pose_cond_tensor = pose_cond_tensor  # (bs, c, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor).unsqueeze(2)
        pose_fea = (
            torch.cat([pose_fea] * 2) if do_classifier_free_guidance else pose_fea
        )

        flag_label_tensor = flag_label.unsqueeze(0).unsqueeze(2).to(
            device=device, dtype=self.pose_guider.dtype
        )
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 3.1 expand the latents if we are doing classifier free guidance


                latent_model_input = torch.cat([latents, flag_label_tensor, image_mask_latents],  dim=1)
                latent_model_input = (
                    torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=clip_vae_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Post-processing
        image = self.vae.decode(latents.squeeze(2) / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.cond_image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        # image = self.decode_latents(latents)  # (b, h, w, c)

        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)

        if not return_dict:
            return image

        return Pose2ImagePipelineOutput(images=image)