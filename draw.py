import cv2
import pickle
import numpy as np
from flag import Flag
import tensorflow as tf
from utils_draw import paint
from utils.gaussian import pdf
import matplotlib.pyplot as plt

flag = Flag()

with open('label.txt', 'r') as f:
    classes = f.readlines()

with open('assets/colors.h5', 'rb') as f:
    colors = pickle.loads(f.read())

with open('dataset/annotations.h5', 'rb') as f:
    annotations = pickle.loads(f.read())

directory = '../COCO/train2017/'
image_name = '000000265725.jpg'
# image_name = '000000000839.jpg'
# image_name = '000000012166.jpg'

for annotation in annotations:
    if image_name == annotation[0]:
        image = cv2.imread(directory + image_name)
        height = annotation[1]
        width = annotation[2]

        label = np.zeros((flag.y_size, flag.y_size, flag.classes), dtype=np.float32)
        x, y, id_ = None, None, None
        objects = []

        for i in range(3, len(annotation)):
            obj = annotation[i]
            id_ = obj['id']
            bbox = obj['bbox']

            if id_ not in objects:
                objects.append(id_)

            image = paint(img=image, bbox=bbox, cls=id_)
            x = tf.range(0, flag.y_size, dtype=tf.float32)
            y = tf.range(0, flag.y_size, dtype=tf.float32)
            x, y = tf.meshgrid(x, y)

            x1 = bbox[0] / flag.x_size * flag.y_size
            y1 = bbox[1] / flag.x_size * flag.y_size
            x2 = bbox[2] / flag.x_size * flag.y_size
            y2 = bbox[3] / flag.x_size * flag.y_size

            xc = (x2 + x1) / 2
            yc = (y2 + y1) / 2
            w = x2 - x1
            h = y2 - y1
            z = pdf(x, xc, w / flag.factor) * pdf(y, yc, h / flag.factor)
            label[:, :, id_] = label[:, :, id_] + z

        label_max = tf.reduce_max(tf.reduce_max(label, axis=0, keepdims=True), axis=1, keepdims=True)
        label = label / label_max

        for id_ in objects:
            # plot label
            category = classes[id_ - 1]
            fig = plt.figure(category)
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, label[:, :, id_], cmap='viridis', edgecolor='none')
            ax.set_title(classes[id_ - 1], fontsize=16)
            ax.view_init(elev=45, azim=60)
            ax.invert_xaxis()
            plt.savefig('figure/' + classes[id_ - 1][:-1] + '.jpg')

        cv2.imwrite('figure/' + image_name, image)
        cv2.imshow(image_name, image)
        plt.show()
        break
