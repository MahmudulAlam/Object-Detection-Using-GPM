import cv2
import pickle
import numpy as np
from flag import Flag
from utils.gaussian import pdf
import matplotlib.pyplot as plt

flag = Flag()

with open('assets/annotations.h5', 'rb') as f:
    annotations = pickle.loads(f.read())

directory = '../COCO/train2017/'
valid_n = 5000
images = np.zeros((valid_n, flag.x_size, flag.x_size, 3), dtype=np.float16)
labels = np.zeros((valid_n, flag.y_size, flag.y_size, flag.classes), dtype=np.float16)

for n, annotation in enumerate(annotations[0:valid_n]):
    print(annotation)
    image = cv2.imread(directory + annotation[0])
    image = cv2.resize(image, (flag.x_size, flag.x_size))
    height = annotation[1]
    width = annotation[2]
    label = np.zeros((flag.y_size, flag.y_size, flag.classes), dtype=np.float16)

    for i in range(3, len(annotation)):
        obj = annotation[i]
        id_ = obj['id']
        bbox = obj['bbox']

        x = np.arange(0, flag.y_size)
        y = np.arange(0, flag.y_size)
        x, y = np.meshgrid(x, y)
        z = np.zeros((flag.y_size, flag.y_size), dtype=np.float16)

        x1 = bbox[0] / flag.x_size * flag.y_size
        y1 = bbox[1] / flag.x_size * flag.y_size
        x2 = bbox[2] / flag.x_size * flag.y_size
        y2 = bbox[3] / flag.x_size * flag.y_size

        xc = (x2 + x1) / 2
        yc = (y2 + y1) / 2
        w = x2 - x1
        h = y2 - y1
        z = z + pdf(x, xc, w / 2) * pdf(y, yc, h / 2)

        label[:, :, id_] = label[:, :, id_] + z

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, z, cmap=plt.get_cmap("coolwarm"), linewidth=0, antialiased=False)

    images[n, :, :, :] = image / 255.
    labels[n, :, :, :] = label / np.max(np.max(label, axis=0, keepdims=True), axis=1, keepdims=True)

labels[np.isnan(labels)] = 0.0

print(images.shape)
print(labels.shape)

np.save('train_x.npy', images)
np.save('train_y.npy', labels)
