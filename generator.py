import cv2
import pickle
import numpy as np
from flag import Flag
from utils.gaussian import pdf
from detect import detector

flag = Flag()

with open('dataset/annotations.h5', 'rb') as f:
    annotations = pickle.loads(f.read())


def batch_indices(batch_size=None, dataset_size=None):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    indices = list(zip(index_a, index_b))
    return indices


def train_generator(batch_size):
    dir = '../COCO/train2017/'
    dataset_size = len(annotations)
    print('Training Dataset Size: ', dataset_size)
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)

    while True:
        np.random.shuffle(annotations)
        for index in indices[:-1]:
            annotation = annotations[index[0]:index[1]]

            images = np.zeros((batch_size, flag.x_size, flag.x_size, 3), dtype=np.float32)
            labels = np.zeros((batch_size, flag.y_size, flag.y_size, flag.classes), dtype=np.float32)

            for n, anno in enumerate(annotation):
                image_name = dir + anno[0]
                # height = anno[1]
                # width = anno[2]
                image = cv2.imread(image_name)
                image = cv2.resize(image, (flag.x_size, flag.x_size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[n, :, :, :] = image
                label = np.zeros((flag.y_size, flag.y_size, flag.classes), dtype=np.float32)

                for i in range(3, len(anno)):
                    obj = anno[i]
                    id_ = obj['id']
                    bbox = obj['bbox']

                    x = np.arange(0, flag.y_size, dtype=np.float32)
                    y = np.arange(0, flag.y_size, dtype=np.float32)
                    x, y = np.meshgrid(x, y)

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

                # label_max = tf.reduce_max(tf.reduce_max(label, axis=0, keepdims=True), axis=1, keepdims=True)
                # label = label / label_max
                label_max = np.max(np.max(label, axis=0, keepdims=True), axis=1, keepdims=True)
                labels[n, :, :, :] = np.divide(label, label_max, out=np.zeros_like(label), where=label_max != 0)

            images = images / 255.
            labels[np.isnan(labels)] = 0.0

            idx = np.arange(images.shape[0])
            np.random.shuffle(idx)
            images = images[idx]
            labels = labels[idx]
            yield images, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open('label.txt', 'r') as f:
        classes = f.readlines()

    gen = train_generator(batch_size=64)
    x_batch, y_batch = next(gen)

    print(x_batch.shape)
    print(y_batch.shape)

    index = 1
    image = x_batch[index]
    label = y_batch[index]

    # image = np.asarray(image * 255., dtype=np.uint8)

    image = detector(image, label)
    cv2.imshow('image', image)

    lmax = np.max(np.max(label, axis=0, keepdims=True), axis=1, keepdims=True)
    lmax = np.squeeze(lmax)
    argmax = np.where(lmax > 0.5)

    for id_ in argmax[0]:
        # plot label
        print(id_)
        x = np.arange(0, flag.y_size, dtype=np.float32)
        y = np.arange(0, flag.y_size, dtype=np.float32)
        x, y = np.meshgrid(x, y)

        category = classes[id_ - 1]
        fig = plt.figure(category)
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, label[:, :, id_], cmap='viridis', edgecolor='none')
        ax.set_title(classes[id_ - 1], fontsize=16)
        ax.view_init(elev=45, azim=60)
        ax.invert_xaxis()
        # plt.savefig('figure/' + classes[id_ - 1][:-1] + '.jpg')

    plt.show()
