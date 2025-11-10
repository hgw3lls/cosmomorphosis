# TileMorph

[![Replicate](https://replicate.com/andreasjansson/tile-morph/badge)](https://replicate.com/andreasjansson/tile-morph)

TileMorph creates a tileable animation between two diffusion prompts. It uses [the circular padding trick](https://gitlab.com/-/snippets/2395088) to generate images that wrap around the edges and now supports the latest [FLUX](https://huggingface.co/black-forest-labs) text-to-image models in addition to traditional Stable Diffusion pipelines.

The animation effect is achieved by interpolating both in CLIP embedding space and latent space.
* The number of CLIP interpolation steps is controlled by the `num_animation_frames` input. Each "animation frame" runs a full Stable Diffusion inference, which makes it slow but interesting.
* The number of latent space interpolation steps between animation frames is controlled by the `num_interpolation_steps` input. Each interpolation step only runs a VAE inference, and is fast but less interesting. You can trade off interestingness versus prediction time by tweaking `num_animation_frames` and `num_interpolation_steps`
* `num_animation_frames * num_interpolation_steps` = number of output frames
* `num_animation_frames * num_interpolation_steps / frames_per_second` = output video length in seconds

You can optionally load LoRA adapters (including FLUX LoRAs) by providing a Hugging Face repository or local path to the `lora_weights` input together with an optional `lora_weight_name` and `lora_scale`.

The predictor now defaults to CPU execution with full `fp32` precision for maximum compatibility. Specify `device=cuda` (and, if desired, a lower precision such as `bf16`) when you want to take advantage of GPU acceleration.

This model supports seamless transitions between different generations. Set `prompt_end` and `seed_end` to the same value of video number _n_ as `prompt_start` and `seed_start` of video number _n + 1_.

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
