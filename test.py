from PIL import Image
from transformers import pipeline

#pipe = pipeline("object-detection")
pipe = pipeline("image-classification")

image=Image.open("./images/2.jpg")

print(pipe(image))
