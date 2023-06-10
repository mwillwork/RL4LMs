import os
import json
import torch
from rl4lms.data_pools.custom_text_generation_pools import DiffusionTextPrompts
from diffusers import StableDiffusionPipeline

########
########
# Purpose of this file is to offline compute all the expert images, which do not change
########
########

def generate_images(better_pipe, starting_idx, texts, image_ids, meta_infos, save_dir, metadata_filename):
    save_images_path = os.path.join(save_dir, 'better_images')
    if not os.path.exists(save_images_path):
        print(f"Had to create path {save_images_path}")
        os.makedirs(save_images_path)

    save_metadata_path = os.path.join(save_dir, metadata_filename)

    better_pipe = better_pipe.to("cuda")
    better_pipeline_output = better_pipe(texts, return_dict=True)
    better_images = better_pipeline_output.images

    if better_pipeline_output.nsfw_content_detected:
         import pdb;pdb.set_trace()
    
    #for idx, bpo in enumerate(better_pipeline_output):
    #    if bpo.nsfw_content_detected:
    #        print(f"Detected NSFW for global idx: {starting_idx + idx}. Will regenerate")
    #        o = better_pipe(texts[idx])
    #        if o.nsfw_content_detected:
    #            raise Exception(f"Two times NSFW dtected for global idx: {starting_idx + idx}. Giving up")
    #        better_images[idx] = o.images[0]

    assert len(better_images) == len(texts)
    all_generation_data = []
    for b_idx, b in enumerate(better_images):
        i_id = image_ids[b_idx]
        better_path = os.path.join(save_images_path, f"{i_id}_better.png")
        b.save(better_path)
        generation_data = {
            "id": i_id,
            "topic": meta_infos[b_idx]["topic"],
            "text_prompt": texts[b_idx],
            "better_image_path": better_path
            }
        all_generation_data.append(generation_data)
        with open(save_metadata_path, "a",) as f:
            f.write(json.dumps(generation_data) + "\n")
    print(f"Saved {len(better_images)} images with generation data: {all_generation_data}")

if __name__ == "__main__":
    better_model_name = "stabilityai/stable-diffusion-2"
    better_pipe = StableDiffusionPipeline.from_pretrained(
            better_model_name, torch_dtype=torch.float16)

    save_dir_stub_path = '/home/ubuntu/RL4LMs/rl4lm_exps/image_generations/'
    starting_index = 0
    # splits = ['val']
    splits = ['train', 'val', 'test']
    for spl in splits:
        dtp_samples = DiffusionTextPrompts.prepare(spl, "NOT_USED")
        dtp_samples = [d[0] for d in dtp_samples]
        save_dir = os.path.join(save_dir_stub_path, spl)
        save_metadata_filename = 'better_offline_metadata.jsonl' 
        text_prompts = [d.prompt_or_input_text for d in dtp_samples]
        image_ids = [d.id for d in dtp_samples]
        meta_infos = [d.meta_data for d in dtp_samples]
        CHUNK_SIZE = 12
        for i in range(starting_index, len(text_prompts), CHUNK_SIZE):
            text_prompts_chunk = text_prompts[i: i + CHUNK_SIZE]
            image_ids_chunk = image_ids[i: i+ CHUNK_SIZE]
            meta_infos_chunk = meta_infos[i: i + CHUNK_SIZE]
   
            print(f"Starting chunk at {i} of {len(text_prompts)} for split {spl}")
            generate_images(better_pipe, i, text_prompts_chunk, image_ids_chunk, meta_infos_chunk, 
                save_dir, save_metadata_filename)


