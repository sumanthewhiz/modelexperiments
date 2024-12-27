import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModelWithProjection
from huggingface_hub import login

login("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")

#model = "../models/sd3_medium.safetensors"

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")

#text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
#pipe = StableDiffusion3Pipeline.from_single_file(model, text_encoder=text_encoder)
pipe.enable_model_cpu_offload()


prompt="A cat holding a sign that says hello world"

#scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,beta_schedule="scaled_linear")

image = pipe(
    prompt,
    negative_prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("./images/output.png")