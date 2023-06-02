
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
import triton_python_backend_utils as pb_utils
from io import BytesIO
import base64
from controlnet_aux import HEDdetector

def get_s3_file(bucket, key):
    s3=boto3.client('s3')
    obj=s3.get_object(
        Bucket=bucket,  
        Key=key,    
    )
    image=obj['Body'].read()
    return image

def put_s3_file(bucket,key,image):
    s3=boto3.client('s3')
    s3.put_object(
        Body=image,
        Bucket=bucket,    
        Key=key   
    )

def _encode(image):
    img = image
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()    
    return img_byte_arr

def _decode(image):    
    image=Image.open(io.BytesIO(image))
    return image


# inference functions ---------------
class TritonPythonModel:
    def initialize(self, args):

        device='cuda'
        self.model_dir = args['model_repository']
        self.model_ver = args['model_version']
  
        controlnet = ControlNetModel.from_pretrained(
            f"{self.model_dir}/{self.model_ver}/hed",
            torch_dtype=torch.float16).to(device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            f"{self.model_dir}/{self.model_ver}/v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16).to(device)
        # change the scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)        
        # enable xformers (optional), requires xformers installation
        self.pipe.enable_xformers_memory_efficient_attention()
        # cpu offload for memory saving, requires accelerate>=0.17.0
        self.pipe.enable_model_cpu_offload()
     

    def execute(self, requests):

        logger = pb_utils.Logger
        responses=[]
        for request in requests:
            input_data=pb_utils.get_input_tensor_by_name(request, "input_payload")
            input_data=json.loads(input_data.as_numpy().item().decode("utf-8"))
            s3_items=input_data['image_uri'].replace('s3://','').split('/',1)    
            bucket=s3_items[0]
            key=s3_items[1]    
            image=get_s3_file(bucket, key)

            # hed Function
            image=_decode(image)
            hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            hed_image = hed(image)            

            seed=int(input_data["seed"])  if "seed" in input_data.keys() else 12345           
            generator = torch.Generator('cuda').manual_seed(seed)  
            
            output_image = self.pipe(
            input_data["prompt"],
            negative_prompt=input_data["negative_prompt"],
            num_inference_steps=int(input_data["steps"])  if "steps" in input_data.keys() else 20,
            generator=generator,
            image=hed_image,
            controlnet_conditioning_scale=float(input_data["scale"])  if "scale" in input_data.keys() else 0.5,
            ).images[0]    
            
            # upload sd-image and hed image to s3
            output = _encode(output_image) 
            output_hed=_encode(hed_image) 
            output_key=Path(input_data['output'],key.split('/')[-1])
            hed_key=Path(input_data['output'],"hed_"+key.split('/')[-1])
            put_s3_file(bucket, str(output_key),output)
            put_s3_file(bucket, str(hed_key),output_hed)
            
            responses.append(pb_utils.InferenceResponse([pb_utils.Tensor("output_image_s3_path", np.array(f"s3://{bucket}/{output_key}").astype(object))]))
        return responses
    
