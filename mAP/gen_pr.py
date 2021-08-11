import os
import cv2
import numpy as np
from flag import Flag
from network import model

flag = Flag()
model = model()
model.load_weights('../weights/weights.h5')


with open('../label.txt', 'r') as f:
    classes = f.readlines()


def create_label(name, img, labels):
    height, width, _ = img.shape
    peaks = np.squeeze(np.max(np.max(labels, axis=0, keepdims=True), axis=1, keepdims=True))
    indices = np.where(peaks > flag.threshold)[0]
    file = open("../../COCO/pr/" + name[:-4] + ".txt", "w")

    for i in indices:
        label = np.asarray(labels[:, :, i], dtype=np.float)
        label[label > flag.threshold] = 255.
        label[label <= flag.threshold] = 0.
        label = np.asarray(label, dtype=np.uint8)

        # smoothening label
        kernel = np.ones((3, 3), np.float32) / 9
        label = cv2.filter2D(label, -1, kernel)

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

            x1 = str(xc - w / 2)
            y1 = str(yc - h / 2)
            x2 = str(xc + w / 2)
            y2 = str(yc + h / 2)

            category = classes[i - 1].replace('\n', '')
            confidence = str(peaks[i])
            text = '{0} {1} {2} {3} {4} {5}\n'.format(category, confidence, x1, y1, x2, y2)
            file.write(text)

    file.close()
    return


def classify(img):
    img = cv2.resize(img, (flag.x_size, flag.x_size))
    img = np.expand_dims(img, axis=0) / 255.
    return model.predict(img)[0]


directory = '../../COCO/val2017/'
image_names = os.listdir(directory)

for n, image_name in enumerate(image_names, 1):
    image = cv2.imread(directory + image_name)
    output = classify(image)
    create_label(image_name, image, output)
    if n % 1000 == 0:
        print('Images: ', n)
