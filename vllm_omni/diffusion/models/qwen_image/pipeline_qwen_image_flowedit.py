# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
    get_qwen_image_edit_plus_post_process_func,
    get_qwen_image_edit_plus_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)

get_qwen_image_flowedit_pre_process_func = get_qwen_image_edit_plus_pre_process_func
get_qwen_image_flowedit_post_process_func = get_qwen_image_edit_plus_post_process_func


class QwenImageFlowEditPipeline(QwenImageEditPlusPipeline):
    """FlowEdit serving pipeline for Qwen-Image-Edit (ICCV 2025).

    Inherits all components from QwenImageEditPlusPipeline. Only overrides
    forward() with a dual-branch differential velocity field loop.

    Input: image=[source_image, condition_image, ...] (>=2 images required)
    - source_image (image[0]): the image being edited -> x_src
    - condition_image(s) (image[1:]): provide context -> VLM encoding + cond_latent
    """

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        image=None,
        true_cfg_scale: float = 7.5,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("This model only supports a single prompt. Taking only the first.")

        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")
        if negative_prompt is None:
            logger.warning(
                "negative_prompt is not set. FlowEdit requires a negative_prompt for CFG. "
                "Using whitespace as default."
            )
            negative_prompt = " "

        if (
            not isinstance(first_prompt, str)
            and "vae_images" in (additional_information := first_prompt.get("additional_information", {}))
            and "condition_images" in additional_information
        ):
            condition_images = additional_information.get("condition_images")
            vae_images = additional_information.get("vae_images")
            vae_image_sizes = additional_information.get("vae_image_sizes")
            height = req.sampling_params.height
            width = req.sampling_params.width
        else:
            raise ValueError(
                "FlowEdit requires preprocessed images via pre_process_func. "
                "Ensure the request contains 'vae_images' and 'condition_images' in additional_information."
            )

        if len(vae_images) < 2:
            if req.is_dummy_run():
                vae_images = vae_images + vae_images
                vae_image_sizes = vae_image_sizes + vae_image_sizes
                condition_images = condition_images + condition_images
            else:
                raise ValueError(
                    "FlowEdit requires at least 2 images: image[0]=source, image[1:]=conditions. "
                    f"Got {len(vae_images)} image(s). Pass the same image twice for single-image editing."
                )

        # FlowEdit parameters (use `is not None` to avoid swallowing falsy values)
        true_cfg_scale_tgt = (
            req.sampling_params.true_cfg_scale
            if req.sampling_params.true_cfg_scale is not None
            else true_cfg_scale
        )
        true_cfg_scale_src = (
            req.sampling_params.true_cfg_scale_src
            if req.sampling_params.true_cfg_scale_src is not None
            else -true_cfg_scale_tgt
        )
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        n_max = (
            req.sampling_params.n_max
            if req.sampling_params.n_max is not None
            else num_inference_steps
        )
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        batch_size = 1

        # VLM encode uses condition images only (excludes source)
        vlm_images = condition_images[1:]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            image=vlm_images,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=negative_prompt,
            image=vlm_images,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            prompt_name="negative_prompt",
        )

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        neg_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

        num_channels_latents = self.transformer.in_channels // 4
        _, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # Split: image[0] -> x_src, image[1:] -> cond_latent
        vw0, vh0 = vae_image_sizes[0]
        seq_len_0 = (vh0 // (self.vae_scale_factor * 2)) * (vw0 // (self.vae_scale_factor * 2))
        x_src = image_latents[:, :seq_len_0, :]
        cond_latent = image_latents[:, seq_len_0:, :]

        noise = torch.randn_like(x_src, generator=generator)

        # img_shapes: [output_shape, *cond_shapes] (no source)
        # Output has same spatial dimensions as source (x_src), so use vae_image_sizes[0]
        img_shapes = [
            [
                (1, vh0 // (self.vae_scale_factor * 2), vw0 // (self.vae_scale_factor * 2)),
                *[
                    (1, vae_height // (self.vae_scale_factor * 2), vae_width // (self.vae_scale_factor * 2))
                    for vae_width, vae_height in vae_image_sizes[1:]
                ],
            ]
        ] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps, sigmas, x_src.shape[1]
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        z_edit = self.flowedit_diffuse(
            prompt_embeds, prompt_embeds_mask, txt_seq_lens,
            negative_prompt_embeds, negative_prompt_embeds_mask, neg_txt_seq_lens,
            x_src, noise, cond_latent,
            img_shapes, timesteps, guidance,
            true_cfg_scale_src, true_cfg_scale_tgt, n_max,
        )

        self._current_timestep = None

        # Use source image dimensions for VAE decode (z_edit matches x_src shape)
        decode_height, decode_width = vh0, vw0
        latents = self._unpack_latents(z_edit, decode_height, decode_width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def flowedit_diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        txt_seq_lens: list[int],
        neg_embeds: torch.Tensor,
        neg_mask: torch.Tensor,
        neg_txt_seq_lens: list[int],
        x_src: torch.Tensor,
        noise: torch.Tensor,
        cond_latent: torch.Tensor,
        img_shapes: list,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
        cfg_scale_src: float,
        cfg_scale_tgt: float,
        n_max: int,
    ) -> torch.Tensor:
        """FlowEdit dual-branch differential loop (2x2 CFG parallel).

        Each branch (src/tgt) calls predict_noise_maybe_with_cfg() which
        handles cond+uncond CFG combine with L2 norm rescale, and
        automatically uses multi-GPU CFG parallel when available.
        """
        num_inference_steps = len(timesteps)
        z_edit = x_src.clone()
        output_slice = x_src.shape[1]

        do_cfg_src = cfg_scale_src != 1.0
        do_cfg_tgt = cfg_scale_tgt != 1.0

        extra_kwargs = {"return_dict": False, "attention_kwargs": self.attention_kwargs}

        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if num_inference_steps - i > n_max:
                    pbar.update()
                    continue

                self._current_timestep = t

                t_curr = t / 1000.0
                t_prev = timesteps[i + 1] / 1000.0 if i < len(timesteps) - 1 else 0.0
                dt = t_prev - t_curr
                timestep = t.expand(x_src.shape[0]).to(x_src.dtype)

                latents_src = (1 - t_curr) * x_src + t_curr * noise
                latents_tgt = z_edit + latents_src - x_src
                model_input_src = torch.cat([latents_src, cond_latent], dim=1)
                model_input_tgt = torch.cat([latents_tgt, cond_latent], dim=1)

                base_pos = {
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_mask": prompt_embeds_mask,
                    "img_shapes": img_shapes,
                    "txt_seq_lens": txt_seq_lens,
                    **extra_kwargs,
                }
                base_neg = {
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": neg_embeds,
                    "encoder_hidden_states_mask": neg_mask,
                    "img_shapes": img_shapes,
                    "txt_seq_lens": neg_txt_seq_lens,
                    **extra_kwargs,
                }

                v_cfg_src = self.predict_noise_maybe_with_cfg(
                    do_cfg_src, cfg_scale_src,
                    {"hidden_states": model_input_src, **base_pos},
                    {"hidden_states": model_input_src, **base_neg},
                    cfg_normalize=True, output_slice=output_slice,
                )

                v_cfg_tgt = self.predict_noise_maybe_with_cfg(
                    do_cfg_tgt, cfg_scale_tgt,
                    {"hidden_states": model_input_tgt, **base_pos},
                    {"hidden_states": model_input_tgt, **base_neg},
                    cfg_normalize=True, output_slice=output_slice,
                )

                z_edit = z_edit + dt * (v_cfg_tgt - v_cfg_src)

                pbar.update()

        return z_edit
