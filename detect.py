import cv2
import pickle
import numpy as np
from flag import Flag

flag = Flag()
with open('assets/colors.h5', 'rb') as f:
    colors = pickle.loads(f.read())

with open('label.txt', 'r') as f:
    classes = f.readlines()


def detector(image, label):
    image = np.asarray(image * 255., np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    indices = np.squeeze(np.max(np.max(label, axis=0, keepdims=True), axis=1, keepdims=True))
    indices = np.where(indices > 0.5)[0]

    for i in indices:
        output = np.asarray(label[:, :, i], dtype=np.float)
        output[output > flag.threshold] = 255.
        output[output <= flag.threshold] = 0.
        output = np.asarray(output, dtype=np.uint8)
        kernel = np.ones((2, 2), np.float32) / 4
        output = cv2.filter2D(output, -1, kernel)
        # cv2.imshow('out', cv2.resize(output, (256, 256)))
        # cv2.waitKey(0)
        _, contours, _ = cv2.findContours(output, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        for contour in contours:
            # print(contour)
            col_wise = contour[:, :, 0]
            row_wise = contour[:, :, 1]

            x1 = min(col_wise)[0] / flag.y_size * flag.x_size
            y1 = min(row_wise)[0] / flag.y_size * flag.x_size
            x2 = max(col_wise)[0] / flag.y_size * flag.x_size
            y2 = max(row_wise)[0] / flag.y_size * flag.x_size
            # print(x1, y1, x2, y2)
            c = colors[i]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (int(c[0]), int(c[1]), int(c[2])), 2)
            # print('class =', classes[i-1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, classes[i - 1][:-1], (int(x1), int(y1)), font, .8, (int(c[0]), int(c[1]), int(c[2])), 2,
                        cv2.LINE_AA)

    return image


if __name__ == '__main__':
    flag = Flag()
    images = np.load('dataset/valid_x.npy')
    labels = np.load('dataset/valid_y.npy')
    # print(images.shape)
    image = images[100]
    label = labels[100]
    image = detector(image, label)
    cv2.imshow('image', image)
    cv2.waitKey(0)
