import os

import cv2
import pickle
import numpy as np
from flag import Flag
from network import model
from PIL import Image, ImageFont, ImageDraw

flag = Flag()
model = model()
model.load_weights('weights/weights_001.h5')

with open('assets/colors.h5', 'rb') as f:
    colors = pickle.loads(f.read())

with open('label.txt', 'r') as f:
    classes = f.readlines()


def paint(img, labels):
    height, width, _ = img.shape

    peaks = np.squeeze(np.max(np.max(labels, axis=0, keepdims=True), axis=1, keepdims=True))
    indices = np.where(peaks > flag.threshold)[0]

    for i in indices:
        label = np.asarray(labels[:, :, i], dtype=np.float)
        label[label > flag.threshold] = 255.
        label[label <= flag.threshold] = 0.
        label = np.asarray(label, dtype=np.uint8)

        # smoothening label
        # kernel = np.ones((3, 3), np.float32) / 9
        # label = cv2.filter2D(label, -1, kernel)

        _, contours, _ = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 10.:
                continue

            col_wise = contour[:, :, 0]
            row_wise = contour[:, :, 1]

            x1 = min(col_wise)[0] / flag.y_size * width
            y1 = min(row_wise)[0] / flag.y_size * height
            x2 = max(col_wise)[0] / flag.y_size * width
            y2 = max(row_wise)[0] / flag.y_size * height

            xc = (x2 + x1) / 2
            yc = (y2 + y1) / 2
            w = x2 - x1
            h = y2 - y1

            w = w * (flag.factor / np.sqrt(2 * 3.1416))
            h = h * (flag.factor / np.sqrt(2 * 3.1416))

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            x2 = x2 if x2 < width else width
            y2 = y2 if y2 < height else height

            img = Image.fromarray(np.uint8(img)).convert('RGB')
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("assets/fonts/Helvetica.ttf", 20)

            # bounding box
            bbox = [x1, y1, x2, y2]
            draw.rectangle(bbox, fill=None, outline=colors[i], width=5)

            # put label on image
            alpha = 10
            category = classes[i - 1]
            text_size = draw.textsize(category, font)
            tx_width, tx_height = text_size
            # print('predicted category: {0:20s}'.format(category), end="\r", flush=True)

            # if bounding box comparatively small than text box then move text box above
            box_width = bbox[2] - bbox[0]
            if tx_width - box_width > -100:
                y1 = bbox[1] - tx_height - alpha
                y2 = bbox[1]
            else:
                y1 = bbox[1]
                y2 = bbox[1] + tx_height + alpha

            x1 = bbox[0]
            x2 = bbox[0] + tx_width + alpha

            draw.rectangle([x1, y1, x2, y2], fill=colors[i], outline=colors[i])
            draw.text([x1 + alpha / 2, y1 + alpha / 2], text=category, fill='white', font=font)

    return img


def classify(img):
    img = cv2.resize(img, (flag.x_size, flag.x_size))
    img = np.expand_dims(img, axis=0) / 255.
    return model.predict(img)[0]


directory = '../COCO/val2017/'

for image_name in os.listdir(directory):
    image = cv2.imread(directory + image_name)
    output = classify(image)
    image = paint(image, output)
    image = np.asarray(image)
    cv2.imwrite('output/' + image_name, image)

