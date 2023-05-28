import torch
from diffusers import StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2"
model_id2 = "CompVis/stable-diffusion-v1-4"
pipe1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe1 = pipe1.to("cuda")

pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16)
pipe2 = pipe2.to("cuda")

def generate_images(p, prompts):
  print(f"About to generate {len(prompts)} images.")
  return p(prompts).images

generate_images(pipe1, ["one prompt", "two prompts", "three prompts", "four prompts"])
generate_images(pipe2, ["one prompt", "two prompts", "three prompts", "four prompts"])

