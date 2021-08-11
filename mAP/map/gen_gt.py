import pickle
from flag import Flag

flag = Flag()

with open('../assets/valid_annotation.h5', 'rb') as f:
    valid_annotation = pickle.loads(f.read())

with open('../label.txt', 'r') as f:
    classes = f.readlines()

print(len(valid_annotation))

for annotation in valid_annotation:
    name = annotation[0][:-4]
    height = annotation[1]
    width = annotation[2]
    file = open("../../COCO/gt/" + name + ".txt", "w")

    for i in range(3, len(annotation)):
        obj = annotation[i]
        id_ = obj['id']
        bbox = obj['bbox']
        category = classes[id_ - 1]

        x1 = str(bbox[0] / flag.x_size * width)
        y1 = str(bbox[1] / flag.x_size * height)
        x2 = str(bbox[2] / flag.x_size * width)
        y2 = str(bbox[3] / flag.x_size * height)

        text = '{0} {1} {2} {3} {4}\n'.format(category.replace('\n', ''), x1, y1, x2, y2)
        file.write(text)

    file.close()
