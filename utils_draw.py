import pickle
import numpy as np
from flag import Flag
from PIL import Image, ImageFont, ImageDraw

flag = Flag()

with open('label.txt', 'r') as f:
    classes = f.readlines()

with open('assets/colors.h5', 'rb') as f:
    colors = pickle.loads(f.read())


def paint(img, bbox, cls):
    height, width, _ = img.shape

    new_bbox = [0, 0, 0, 0]
    new_bbox[0] = bbox[0] / flag.x_size * width
    new_bbox[1] = bbox[1] / flag.x_size * height
    new_bbox[2] = bbox[2] / flag.x_size * width
    new_bbox[3] = bbox[3] / flag.x_size * height

    img = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("assets/fonts/Helvetica.ttf", 20)

    # bounding box
    draw.rectangle(new_bbox, fill=None, outline=colors[cls], width=5)

    # put label on image
    alpha = 10
    category = classes[cls - 1]
    text_size = draw.textsize(category, font)
    tx_width, tx_height = text_size

    # if bounding box comparatively small than text box then move text box above
    box_width = new_bbox[2] - new_bbox[0]
    if tx_width - box_width > -100:
        y1 = new_bbox[1] - tx_height - alpha
        y2 = new_bbox[1]
    else:
        y1 = new_bbox[1]
        y2 = new_bbox[1] + tx_height + alpha

    x1 = new_bbox[0]
    x2 = new_bbox[0] + tx_width + alpha

    draw.rectangle([x1, y1, x2, y2], fill=colors[cls], outline=colors[cls])
    draw.text([x1 + alpha / 2, y1 + alpha / 2], text=category, fill='white', font=font)
    img = np.asarray(img)
    return img
