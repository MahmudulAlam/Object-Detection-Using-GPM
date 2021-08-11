import pickle
from flag import Flag
from pycocotools.coco import COCO

flag = Flag()
coco = COCO('../COCO/annotations/instances_val2017.json')

labels = []

for image_id in coco.getImgIds():
    image = coco.loadImgs(image_id)[0]
    anno_ids = coco.getAnnIds(image_id)
    annotations = coco.loadAnns(anno_ids)

    image_name = image.get('file_name')
    img_height = image.get('height')
    img_width = image.get('width')
    label = [image_name, img_height, img_width]

    objects_in_image = {}
    for annotation in annotations:
        bbox = annotation.get('bbox')
        x1, y1, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        x2, y2 = x1 + width, y1 + height
        bbox = [x1, y1, x2, y2]

        bbox = [bbox[0] / img_width * flag.x_size,
                bbox[1] / img_height * flag.x_size,
                bbox[2] / img_width * flag.x_size,
                bbox[3] / img_height * flag.x_size]

        objects_in_image = {'id': annotation.get('category_id'), 'bbox': bbox}
        label.append(objects_in_image)

    labels.append(label)

print('Dataset Size:', len(labels))

with open('valid_annotation.h5', 'wb') as f:
    pickle.dump(labels, f)
