import os, json, requests, runpod

import torch
import random
import math
import time
from typing import Union, List
import PIL.Image
from datetime import datetime
import numpy as np
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler,CogVideoXDPMScheduler
import moviepy.editor as mp

def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path

def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"{timestamp}.mp4"
    export_to_video(tensor, video_path, fps=fps)
    return video_path

with torch.inference_mode():
    pipe = CogVideoXPipeline.from_pretrained("/content/model", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.transformer.to(memory_format=torch.channels_last)
    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    prompt = values['prompt']
    seed = values['seed']
    width = values['width']
    height = values['height']
    num_inference_steps = values['num_inference_steps']
    num_frames = values['num_frames']
    use_dynamic_cfg = values['use_dynamic_cfg']
    guidance_scale = values['guidance_scale']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    video_pt = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=use_dynamic_cfg,
        output_type="pt",
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).frames

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    video_path = save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))
    # gif_path = convert_to_gif(video_path)

    result = video_path
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})