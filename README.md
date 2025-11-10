# TileMorph

[![Replicate](https://replicate.com/andreasjansson/tile-morph/badge)](https://replicate.com/andreasjansson/tile-morph)

TileMorph creates a tileable animation between two diffusion prompts. It uses [the circular padding trick](https://gitlab.com/-/snippets/2395088) to generate images that wrap around the edges and now supports the latest [FLUX](https://huggingface.co/black-forest-labs) text-to-image models in addition to traditional Stable Diffusion pipelines.

## How the animation pipeline works

1. **Prompt encoding** – Both `prompt_start` and `prompt_end` are converted into text embeddings (and, if requested, LoRA weights are merged into the text encoder). These embeddings are cached so the same conditioning can be reused across animation frames.
2. **Anchor denoising** – We run the chosen diffusion pipeline (Stable Diffusion or FLUX) once per key prompt to generate "anchor" latents that define the endpoints of the animation. Seeds keep these anchors repeatable.
3. **Prompt & noise blending** – For each intermediate animation frame we slerp both the text embeddings and the random noise, then run another denoising pass. `num_animation_frames` controls how many of these heavy diffusion steps happen between each pair of prompts.
4. **Latent interpolation & video assembly** – Between consecutive denoised latents we optionally linearly blend `num_interpolation_steps` intermediate tensors, decode them with the VAE, and stitch the resulting images into an MP4 at `frames_per_second`.

The predictor automatically switches between CPU and GPU math depending on the `device` input, defaulting to `cpu` for the broadest compatibility.

The animation effect is achieved by interpolating both in CLIP embedding space and latent space.
* The number of CLIP interpolation steps is controlled by the `num_animation_frames` input. Each "animation frame" runs a full Stable Diffusion inference, which makes it slow but interesting.
* The number of latent space interpolation steps between animation frames is controlled by the `num_interpolation_steps` input. Each interpolation step only runs a VAE inference, and is fast but less interesting. You can trade off interestingness versus prediction time by tweaking `num_animation_frames` and `num_interpolation_steps`
* `num_animation_frames * num_interpolation_steps` = number of output frames
* `num_animation_frames * num_interpolation_steps / frames_per_second` = output video length in seconds

You can optionally load LoRA adapters (including FLUX LoRAs) by providing a Hugging Face repository or local path to the `lora_weights` input together with an optional `lora_weight_name` and `lora_scale`.

The predictor now defaults to CPU execution with full `fp32` precision for maximum compatibility. Specify `device=cuda` (and, if desired, a lower precision such as `bf16`) when you want to take advantage of GPU acceleration.

This model supports seamless transitions between different generations. Set `prompt_end` and `seed_end` to the same value of video number _n_ as `prompt_start` and `seed_start` of video number _n + 1_.

When you want to anchor either side of the animation to an existing asset, provide the optional `start_image` and/or `end_image` inputs. TileMorph will encode those images into the latent space so the first and/or last animation frames match your references before transitioning through the rest of the prompts. You can provide just one of the images (or neither) depending on which side of the animation you want to lock down.

For example, this command only anchors the first frame and lets the animation freely discover the ending:

```bash
cog predict \
    -i prompt_start="underwater kelp forest" \
    -i prompt_end="nebula made of seaweed" \
    -i start_image=@./reference/start.png
```

To do the opposite—fixing only the last frame—swap `start_image` for `end_image` in the invocation above.

## macOS setup and usage

Apple Silicon and Intel Macs can run TileMorph entirely on the CPU. The following steps assume a clean macOS 13+ installation:

1. **Install system tooling**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.10 ffmpeg git
   pip3 install --user cog
   ```
   Restart your shell (or follow the Homebrew post-install instructions) so that `brew`, `python3`, and `cog` are on your `$PATH`.
2. **Create an isolated environment**
   ```bash
   python3 -m venv ~/.venvs/tilemorph
   source ~/.venvs/tilemorph/bin/activate
   pip install --upgrade pip
   pip install cog diffusers transformers accelerate safetensors opencv-python torch torchvision torchaudio
   ```
   The universal PyTorch wheels run natively on Apple Silicon; if you prefer Intel-only binaries you can append `--index-url https://download.pytorch.org/whl/cpu` to the final command.
3. **Download model weights**
   ```bash
   cog run script/download-weights <your-hugging-face-auth-token>
   ```
   You can omit the token if you have already accepted the relevant model licenses on Hugging Face.
4. **Run a local prediction**
   ```bash
   cog predict \
       -i prompt_start="sunlit coral reef" \
       -i prompt_end="aurora over snowy mountains" \
       -i seed_start=123 \
       -i seed_end=456 \
       -i base_model="black-forest-labs/FLUX.1-dev" \
       -i device=cpu
   ```
   The default `device=cpu` works well on macOS. If you have an external GPU enclosure that exposes CUDA, you can override the device with `-i device=cuda` to enable mixed precision.

## Development

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens). (Leave the token blank if you have already accepted the model license and are using a locally cached copy.):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict \
        -i prompt_start="colorful abstract patterns" \
        -i prompt_end="tropical jungle, cgsociety" \
        -i seed_start=1 \
        -i seed_end=2 \
        -i device=cuda \
        -i base_model="black-forest-labs/FLUX.1-dev" \
        -i lora_weights="black-forest-labs/FLUX.1-dev-lora-watercolor" \
        -i lora_scale=0.6

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
