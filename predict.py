from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import av
import cv2
import numpy as np
import os
import torch
from PIL import Image
from diffusers import (
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from cog import BasePredictor, Input, Path

MODEL_CACHE = "diffusers-cache"


@dataclass
class FluxEmbeddings:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


@contextmanager
def lora_adapter(
    pipe,
    lora_weights: Optional[str],
    lora_scale: Optional[float] = None,
    weight_name: Optional[str] = None,
    adapter_name: str = "_tmp_lora",
):
    if not lora_weights:
        yield
        return

    if not hasattr(pipe, "load_lora_weights"):
        raise ValueError("This pipeline does not support loading LoRA adapters.")

    load_kwargs = {"adapter_name": adapter_name}
    if weight_name:
        load_kwargs["weight_name"] = weight_name
    pipe.load_lora_weights(lora_weights, **load_kwargs)

    fuse_kwargs = {}
    if lora_scale is not None:
        fuse_kwargs["lora_scale"] = lora_scale
    pipe.fuse_lora(**fuse_kwargs)

    try:
        yield
    finally:
        if hasattr(pipe, "unfuse_lora"):
            pipe.unfuse_lora()
        if hasattr(pipe, "delete_adapters"):
            pipe.delete_adapters([adapter_name])
        elif hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()


def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        for k, v in patch.items():
            kwargs[k] = v
        return init(self, *args, **kwargs)

    cls.__init__ = __init__


patch_conv(padding_mode="circular")


class Predictor(BasePredictor):
    def setup(self):
        print("Initializing predictor caches...")
        self.device = "cpu"
        self._pipelines: dict[Tuple[str, str, str, str], Tuple[Any, str]] = {}

    def get_pipeline(
        self,
        base_model: str,
        precision: str,
        scheduler_name: str,
        device: str,
    ) -> Tuple[Any, str]:
        precision = precision.lower()
        key = (base_model, precision, scheduler_name, device)
        if key in self._pipelines:
            return self._pipelines[key]
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported precision '{precision}'.")
        dtype = dtype_map[precision]

        if device.startswith("cpu") and precision != "fp32":
            raise ValueError("CPU execution only supports fp32 precision.")

        is_flux = "flux" in base_model.lower()

        if is_flux:
            pipe: FluxPipeline = FluxPipeline.from_pretrained(
                base_model,
                torch_dtype=dtype,
                cache_dir=MODEL_CACHE,
                local_files_only=False,
            )
            pipe.to(device, dtype=dtype)
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
            backend = "flux"
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=dtype,
                cache_dir=MODEL_CACHE,
                local_files_only=False,
            )
            pipe.to(device, dtype=dtype)
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
            pipe.scheduler = make_scheduler(pipe, scheduler_name)
            backend = "stable"

        pipe.set_progress_bar_config(disable=True)
        self._pipelines[key] = (pipe, backend)
        return pipe, backend

    def _resolve_device(self, device_choice: str) -> str:
        choice = device_choice.lower()
        if choice == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device was requested but is not available in this environment."
                )
            return "cuda"
        if choice != "cpu":
            raise ValueError(
                "Device must be one of 'cpu', 'cuda', or 'auto'."
            )
        return "cpu"

    def predict(
        self,
        prompt_start: str = Input(description="Prompt to start the animation with"),
        prompt_end: str = Input(
            description="Prompt to end the animation with. You can include multiple prompts by separating the prompts with | (the 'pipe' character)"
        ),
        width: int = Input(
            description="Width of output video",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output video",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        num_interpolation_steps: int = Input(
            description="Number of steps to interpolate between animation frames",
            ge=0,
            le=1000,
            default=20,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=5000, default=30
        ),
        num_animation_frames: int = Input(
            description="Number of frames to animate", default=10, ge=2, le=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=3.5
        ),
        frames_per_second: int = Input(
            description="Frames per second in output video",
            default=20,
            ge=1,
            le=60,
        ),
        intermediate_output: bool = Input(
            description="Whether to display intermediate outputs during generation",
            default=False,
        ),
        seed_start: int = Input(
            description="Random seed for first prompt. Leave blank to randomize the seed",
            default=None,
        ),
        seed_end: int = Input(
            description="Random seed for last prompt. Leave blank to randomize the seed",
            default=None,
        ),
        base_model: str = Input(
            description="Base diffusion model to load",
            default="black-forest-labs/FLUX.1-dev",
        ),
        precision: str = Input(
            description="Precision to run the model with",
            default="fp32",
            choices=["fp16", "bf16", "fp32"],
        ),
        scheduler: str = Input(
            description="Scheduler to use with Stable Diffusion backends",
            default="euler",
            choices=["pndm", "lms", "euler"],
        ),
        device: str = Input(
            description="Device to run inference on (defaults to CPU)",
            default="cpu",
            choices=["cpu", "cuda", "auto"],
        ),
        lora_weights: Optional[str] = Input(
            description="Optional path or Hugging Face repo containing a LoRA adapter",
            default=None,
        ),
        lora_scale: float = Input(
            description="Scale factor to apply to the loaded LoRA",
            default=0.75,
        ),
        lora_weight_name: Optional[str] = Input(
            description="Optional specific weight filename within the LoRA repo",
            default=None,
        ),
        start_image: Optional[Path] = Input(
            description="Optional initial image to seed the first animation frame",
            default=None,
        ),
        end_image: Optional[Path] = Input(
            description="Optional final image to use for the last animation frame",
            default=None,
        ),
    ) -> Iterator[Path]:
        if seed_start is None:
            seed_start = int.from_bytes(os.urandom(2), "big")
        if seed_end is None:
            seed_end = int.from_bytes(os.urandom(2), "big")
        print(f"Using seeds: {seed_start}, {seed_end}")

        resolved_device = self._resolve_device(device)
        self.device = resolved_device
        pipe, backend = self.get_pipeline(
            base_model, precision, scheduler, resolved_device
        )

        autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[precision.lower()]
        autocast_device = (
            "cuda" if self.device.startswith("cuda") else self.device
        )

        prompts = [prompt_start] + [
            p.strip() for p in prompt_end.split("|") if p.strip()
        ]

        with (
            torch.autocast(autocast_device, dtype=autocast_dtype)
            if self.device != "cpu"
            else nullcontext()
        ), torch.inference_mode(), lora_adapter(
            pipe, lora_weights, lora_scale, lora_weight_name
        ):
            if backend == "flux":
                yield from self._run_flux(
                    pipe,
                    prompts,
                    width,
                    height,
                    num_interpolation_steps,
                    num_inference_steps,
                    num_animation_frames,
                    guidance_scale,
                    frames_per_second,
                    intermediate_output,
                    seed_start,
                    seed_end,
                    start_image,
                    end_image,
                )
            else:
                yield from self._run_stable(
                    pipe,
                    prompts,
                    width,
                    height,
                    num_interpolation_steps,
                    num_inference_steps,
                    num_animation_frames,
                    guidance_scale,
                    frames_per_second,
                    intermediate_output,
                    seed_start,
                    seed_end,
                    scheduler,
                    start_image,
                    end_image,
                )

    def _run_stable(
        self,
        pipe: StableDiffusionPipeline,
        prompts: List[str],
        width: int,
        height: int,
        num_interpolation_steps: int,
        num_inference_steps: int,
        num_animation_frames: int,
        guidance_scale: float,
        frames_per_second: int,
        intermediate_output: bool,
        seed_start: int,
        seed_end: int,
        scheduler_name: str,
        start_image: Optional[Path],
        end_image: Optional[Path],
    ) -> Iterator[Path]:
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator_start = torch.Generator(device=generator_device).manual_seed(
            seed_start
        )
        generator_end = torch.Generator(device=generator_device).manual_seed(seed_end)

        pipe.scheduler = make_scheduler(pipe, scheduler_name)
        noise_latents_start = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator_start,
            device=self.device,
            dtype=pipe.unet.dtype,
        )
        noise_latents_end = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator_end,
            device=self.device,
            dtype=pipe.unet.dtype,
        )

        noise_latents_start_cpu = noise_latents_start.detach().to("cpu", torch.float32)
        noise_latents_end_cpu = noise_latents_end.detach().to("cpu", torch.float32)

        do_cfg = guidance_scale > 1.0
        keyframe_embeddings: List[torch.Tensor] = []
        for prompt in prompts:
            keyframe_embeddings.append(
                pipe._encode_prompt(prompt, self.device, 1, do_cfg, "")
            )

        if start_image:
            latents_start_gpu = self._encode_stable_image(
                pipe, start_image, width, height
            )
        else:
            pipe.scheduler = make_scheduler(pipe, scheduler_name)
            latents_start_gpu = self.denoise(
                pipe,
                noise_latents_start,
                keyframe_embeddings[0],
                num_inference_steps,
                guidance_scale,
                generator_start,
            )
        image_start = pipe.decode_latents(latents_start_gpu)
        pipe.run_safety_checker(image_start, self.device, keyframe_embeddings[0].dtype)
        latents_start = latents_start_gpu.detach().to("cpu", torch.float32)

        if end_image:
            latents_end_gpu = self._encode_stable_image(
                pipe, end_image, width, height
            )
        else:
            pipe.scheduler = make_scheduler(pipe, scheduler_name)
            latents_end_gpu = self.denoise(
                pipe,
                noise_latents_end,
                keyframe_embeddings[-1],
                num_inference_steps,
                guidance_scale,
                generator_end,
            )
        image_end = pipe.decode_latents(latents_end_gpu)
        pipe.run_safety_checker(image_end, self.device, keyframe_embeddings[-1].dtype)
        latents_end = latents_end_gpu.detach().to("cpu", torch.float32)

        frames_latents: List[torch.Tensor] = []

        if intermediate_output:
            yield save_pil_image(
                pipe.numpy_to_pil([image_start])[0], path="/tmp/output-0.png"
            )

        for keyframe in range(len(prompts) - 1):
            for i in range(num_animation_frames):
                if keyframe == 0 and i == 0:
                    latents_gpu = latents_start_gpu
                    latents_cpu = latents_start
                else:
                    ratio = i / num_animation_frames
                    text_embeddings = slerp(
                        ratio,
                        keyframe_embeddings[keyframe],
                        keyframe_embeddings[keyframe + 1],
                    )
                    noise_latents = slerp(
                        ratio, noise_latents_start_cpu, noise_latents_end_cpu
                    ).to(self.device, dtype=pipe.unet.dtype)

                    pipe.scheduler = make_scheduler(pipe, scheduler_name)
                    latents_gpu = self.denoise(
                        pipe,
                        noise_latents,
                        text_embeddings,
                        num_inference_steps,
                        guidance_scale,
                        generator_start,
                    )
                    latents_cpu = latents_gpu.detach().to("cpu", torch.float32)

                frames_latents.append(latents_cpu)

                if intermediate_output and (i > 0 or keyframe > 0):
                    image = pipe.decode_latents(latents_gpu)
                    yield save_pil_image(
                        pipe.numpy_to_pil([image])[0],
                        path=f"/tmp/output-{keyframe}-{i}.png",
                    )

        frames_latents.append(latents_end)

        def decode_latents_fn(latents: torch.Tensor) -> np.ndarray:
            latents = latents.to(self.device, dtype=pipe.unet.dtype)
            image = pipe.decode_latents(latents)[0].astype("float32")
            return image

        images = self.interpolate_latents(
            frames_latents, num_interpolation_steps, decode_latents_fn
        )
        yield self.save_mp4(images, frames_per_second, width, height)

    def _run_flux(
        self,
        pipe: FluxPipeline,
        prompts: List[str],
        width: int,
        height: int,
        num_interpolation_steps: int,
        num_inference_steps: int,
        num_animation_frames: int,
        guidance_scale: float,
        frames_per_second: int,
        intermediate_output: bool,
        seed_start: int,
        seed_end: int,
        start_image: Optional[Path],
        end_image: Optional[Path],
    ) -> Iterator[Path]:
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator_start = torch.Generator(device=generator_device).manual_seed(
            seed_start
        )
        generator_end = torch.Generator(device=generator_device).manual_seed(seed_end)

        noise_latents_start = self._flux_noise_latents(
            pipe, height, width, generator_start
        )
        noise_latents_end = self._flux_noise_latents(pipe, height, width, generator_end)

        noise_latents_start_cpu = noise_latents_start.detach().to("cpu", torch.float32)
        noise_latents_end_cpu = noise_latents_end.detach().to("cpu", torch.float32)

        keyframe_embeddings: List[FluxEmbeddings] = []
        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
            )
            keyframe_embeddings.append(
                FluxEmbeddings(prompt_embeds, pooled_prompt_embeds)
            )

        if start_image:
            latents_start_gpu = self._encode_flux_image(
                pipe, start_image, width, height
            )
        else:
            latents_start_gpu = self._flux_denoise(
                pipe,
                noise_latents_start,
                keyframe_embeddings[0],
                num_inference_steps,
                guidance_scale,
                generator_start,
                height,
                width,
            )
        image_start = self._decode_flux_latents(
            pipe, latents_start_gpu, height, width
        )
        latents_start = latents_start_gpu.detach().to("cpu", torch.float32)

        if end_image:
            latents_end_gpu = self._encode_flux_image(
                pipe, end_image, width, height
            )
        else:
            latents_end_gpu = self._flux_denoise(
                pipe,
                noise_latents_end,
                keyframe_embeddings[-1],
                num_inference_steps,
                guidance_scale,
                generator_end,
                height,
                width,
            )
        image_end = self._decode_flux_latents(pipe, latents_end_gpu, height, width)
        latents_end = latents_end_gpu.detach().to("cpu", torch.float32)

        frames_latents: List[torch.Tensor] = []

        if intermediate_output:
            yield save_pil_image(
                pipe.numpy_to_pil([image_start])[0], path="/tmp/output-0.png"
            )

        for keyframe in range(len(prompts) - 1):
            for i in range(num_animation_frames):
                if keyframe == 0 and i == 0:
                    latents_gpu = latents_start_gpu
                    latents_cpu = latents_start
                else:
                    ratio = i / num_animation_frames
                    prompt_embeds = slerp(
                        ratio,
                        keyframe_embeddings[keyframe].prompt_embeds,
                        keyframe_embeddings[keyframe + 1].prompt_embeds,
                    )
                    pooled_prompt_embeds = slerp(
                        ratio,
                        keyframe_embeddings[keyframe].pooled_prompt_embeds,
                        keyframe_embeddings[keyframe + 1].pooled_prompt_embeds,
                    )
                    embeddings = FluxEmbeddings(
                        prompt_embeds.to(self.device, dtype=pipe.transformer.dtype),
                        pooled_prompt_embeds.to(
                            self.device, dtype=pipe.transformer.dtype
                        ),
                    )

                    noise_latents = slerp(
                        ratio, noise_latents_start_cpu, noise_latents_end_cpu
                    ).to(self.device, dtype=pipe.transformer.dtype)

                    latents_gpu = self._flux_denoise(
                        pipe,
                        noise_latents,
                        embeddings,
                        num_inference_steps,
                        guidance_scale,
                        generator_start,
                        height,
                        width,
                    )
                    latents_cpu = latents_gpu.detach().to("cpu", torch.float32)

                frames_latents.append(latents_cpu)

                if intermediate_output and (i > 0 or keyframe > 0):
                    image = self._decode_flux_latents(
                        pipe, latents_gpu, height, width
                    )
                    yield save_pil_image(
                        pipe.numpy_to_pil([image])[0],
                        path=f"/tmp/output-{keyframe}-{i}.png",
                    )

        frames_latents.append(latents_end)

        def decode_latents_fn(latents: torch.Tensor) -> np.ndarray:
            return self._decode_flux_latents(pipe, latents, height, width)

        images = self.interpolate_latents(
            frames_latents, num_interpolation_steps, decode_latents_fn
        )
        yield self.save_mp4(images, frames_per_second, width, height)

    def _encode_stable_image(
        self,
        pipe: StableDiffusionPipeline,
        image_path: Path,
        width: int,
        height: int,
    ) -> torch.Tensor:
        image = self._load_image(image_path, width, height)
        pixel_values = pipe.image_processor.preprocess(image)
        pixel_values = pixel_values.to(self.device, dtype=pipe.vae.dtype)
        with torch.inference_mode():
            latents = pipe.vae.encode(pixel_values).latent_dist.mode()
        latents = latents * pipe.vae.config.scaling_factor
        return latents.to(self.device, dtype=pipe.unet.dtype)

    def _encode_flux_image(
        self,
        pipe: FluxPipeline,
        image_path: Path,
        width: int,
        height: int,
    ) -> torch.Tensor:
        image = self._load_image(image_path, width, height)
        pixel_values = pipe.image_processor.preprocess(image)
        pixel_values = pixel_values.to(self.device, dtype=pipe.vae.dtype)
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
        if torch.is_tensor(shift):
            shift_tensor = shift.to(self.device, dtype=pixel_values.dtype)
        elif isinstance(shift, (tuple, list)):
            shift_tensor = torch.tensor(shift, device=self.device, dtype=pixel_values.dtype).view(1, -1, 1, 1)
        else:
            shift_tensor = torch.tensor(float(shift), device=self.device, dtype=pixel_values.dtype)
        pixel_values = pixel_values - shift_tensor
        with torch.inference_mode():
            latents = pipe.vae.encode(pixel_values).latent_dist.mode()
        if torch.is_tensor(scale):
            scale_tensor = scale.to(self.device, dtype=latents.dtype)
        elif isinstance(scale, (tuple, list)):
            scale_tensor = torch.tensor(scale, device=self.device, dtype=latents.dtype).view(1, -1, 1, 1)
        else:
            scale_tensor = torch.tensor(float(scale), device=self.device, dtype=latents.dtype)
        latents = latents * scale_tensor
        num_channels = pipe.transformer.config.in_channels // 4
        latent_height = 2 * (int(height) // pipe.vae_scale_factor)
        latent_width = 2 * (int(width) // pipe.vae_scale_factor)
        latents = pipe._pack_latents(latents, 1, num_channels, latent_height, latent_width)
        return latents.to(self.device, dtype=pipe.transformer.dtype)

    def _load_image(self, image_path: Path, width: int, height: int) -> Image.Image:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.LANCZOS)
            image = image.copy()
        return image

    def _flux_noise_latents(
        self,
        pipe: FluxPipeline,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        latent_height = 2 * (int(height) // pipe.vae_scale_factor)
        latent_width = 2 * (int(width) // pipe.vae_scale_factor)
        num_channels = pipe.transformer.config.in_channels // 4
        latents = randn_tensor(
            (1, num_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=pipe.transformer.dtype,
        )
        return pipe._pack_latents(
            latents, 1, num_channels, latent_height, latent_width
        )

    def _flux_denoise(
        self,
        pipe: FluxPipeline,
        latents: torch.Tensor,
        embeddings: FluxEmbeddings,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        height: int,
        width: int,
    ) -> torch.Tensor:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        prompt_embeds = embeddings.prompt_embeds.to(
            self.device, dtype=pipe.transformer.dtype
        )
        pooled_prompt_embeds = embeddings.pooled_prompt_embeds.to(
            self.device, dtype=pipe.transformer.dtype
        )
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=latents,
            height=height,
            width=width,
            output_type="latent",
        )
        return result.images

    def _decode_flux_latents(
        self,
        pipe: FluxPipeline,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> np.ndarray:
        latents = latents.to(self.device, dtype=pipe.transformer.dtype)
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.inference_mode():
            image = pipe.vae.decode(latents, return_dict=False)[0]
        images = pipe.image_processor.postprocess(image, output_type="np")
        if isinstance(images, list):
            images = images[0]
        return images.astype(np.float32)

    def interpolate_latents(
        self,
        frames_latents: List[torch.Tensor],
        num_interpolation_steps: int,
        decode_fn,
    ) -> List[np.ndarray]:
        print("Interpolating images from latents")
        images: List[np.ndarray] = []
        with torch.inference_mode():
            if num_interpolation_steps <= 0:
                for latents in frames_latents:
                    images.append(decode_fn(latents))
                return images

            for i in range(len(frames_latents) - 1):
                latents_start = frames_latents[i]
                latents_end = frames_latents[i + 1]
                for j in range(num_interpolation_steps):
                    x = j / num_interpolation_steps
                    latents = latents_start * (1 - x) + latents_end * x
                    images.append(decode_fn(latents))
        return images

    def save_mp4(
        self, images: List[np.ndarray], fps: int, width: int, height: int
    ) -> Path:
        print("Saving MP4")
        output_path = "/tmp/output.mp4"

        output = av.open(output_path, "w")
        stream = output.add_stream(
            "h264",
            rate=fps,
            options={
                "crf": "10",
                "tune": "film",
            },
        )
        stream.width = width
        stream.height = height

        for image in images:
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            packet = stream.encode(frame)
            output.mux(packet)

        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        return Path(output_path)

    def denoise(
        self,
        pipe: StableDiffusionPipeline,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        eta = 0
        timesteps = pipe.scheduler.timesteps
        do_cfg = guidance_scale > 1.0

        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = (
                    torch.cat([latents] * 2) if do_cfg else latents
                )
                latent_model_input = pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = pipe.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                progress_bar.update()

        return latents


def make_scheduler(pipe: StableDiffusionPipeline, scheduler_name: str):
    name = scheduler_name.lower()
    scheduler_map = {
        "pndm": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
    }
    scheduler_cls = scheduler_map.get(name, EulerDiscreteScheduler)
    return scheduler_cls.from_config(pipe.scheduler.config)


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
    else:
        inputs_are_torch = False

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def save_pil_image(image, path):
    image.save(path)
    return Path(path)
