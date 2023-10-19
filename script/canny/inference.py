import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
import io
import base64
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler
import boto3
from pathlib import Path


def _encode(image):
    img = image
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_byte_arr = base64.b64encode(img_byte_arr).decode()
    return img_byte_arr

def _decode(image):
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    return image


# inference functions ---------------
def model_fn(model_dir):

    control_net_postfix=[
            "canny",
            "depth",
            "hed",
            "mlsd",
            "openpose",
            "scribble"
        ]

    control_net=[x for x in os.listdir(model_dir) if x in control_net_postfix][0]

    # Creating pipeline model
    controlnet = ControlNetModel.from_pretrained(
        f"{model_dir}/{control_net}",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"{model_dir}/v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # change the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        # enable xformers (optional), requires xformers installation
        pipe.enable_xformers_memory_efficient_attention()
        # cpu offload for memory saving, requires accelerate>=0.17.0
        pipe.enable_model_cpu_offload()
    return pipe

def transform_fn(model, data, input_content_type, output_content_type):

    input_data = json.loads(data)

    # Canny Function
    image = _decode(input_data['image'])
    image = np.array(image)
    low_threshold = int(input_data["low_threshold"])  if "low_threshold" in input_data.keys() else 100
    high_threshold = int(input_data["high_threshold"]) if "high_threshold" in input_data.keys() else 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    seed = int(input_data["seed"])  if "seed" in input_data.keys() else 12345
    if torch.cuda.is_available():
        generator = torch.Generator('cuda').manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    # Generating output image
    output_image = model(
        input_data["prompt"],
        negative_prompt=input_data["negative_prompt"],
        num_inference_steps=int(input_data["steps"])  if "steps" in input_data.keys() else 20,
        generator=generator,
        image=canny_image,
        controlnet_conditioning_scale=float(input_data["scale"])  if "scale" in input_data.keys() else 0.5,
    ).images[0]

    output = _encode(output_image)
    output_canny = _encode(canny_image)

    # Returning the output image and canny edge image
    response = {
        "output_image":output,
        "canny_image":output_canny
    }
    return response