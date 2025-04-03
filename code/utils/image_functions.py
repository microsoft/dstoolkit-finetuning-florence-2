from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# create functions for reading and displaying images
def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image

def display_image(image, title=None):
    plt.imshow(np.array(image))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()