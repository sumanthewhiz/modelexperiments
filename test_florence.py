import requests
import torch
#from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

from PIL import Image, ImageDraw, ImageFont
import requests
import copy
import torch
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bbox(image, data):

    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.show()
    


#try:
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

#Object Detection
#prompt = "<OD>"

#OCR
#prompt="<OCR>"

#Caption
#prompt = "<CAPTION>"

#Dense Region Caption
prompt="<DENSE_REGION_CAPTION>"

#Detailed Caption
#prompt = "<DETAILED_CAPTION>"

#More Detailed Caption
#prompt = "<MORE_DETAILED_CAPTION>"

#Caption to Phrase Grounding (caption to phrase grounding task requires additional text input, i.e. caption)
#prompt = "<CAPTION_TO_PHRASE_GROUNDING>"


image = Image.open("./images/3.jpg")

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

#print("------------------------------------------------------------")
#print(generated_ids)

print("------------------------------------------------------------")
print(parsed_answer)
print("------------------------------------------------------------")

draw_bbox(image, parsed_answer[f'{prompt}'])

#except Exception:
#    pass