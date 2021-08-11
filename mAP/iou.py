import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

with open('output/output.txt', "r") as f:
    lines = f.readlines()

iou = []
classes = []

lines = lines[1:]

for line in lines:
    if line[0] == '#':
        break
    if line[0] == '\n':
        continue
    line = line.replace('=', ':').strip().split(':')
    if line[-1][-2:] == 'AP':
        classes.append(line[-1][:-2])

    if line[0] == 'IOU ':
        value = line[1]
        if value == ' nan':
            value = 0.0
        else:
            value = float(value)
        iou.append(value)

iou = np.array(iou)
classes = np.array(classes)

arg = np.argsort(iou)
print(arg)
iou = iou[arg]
classes = classes[arg]
plt.figure(figsize=(9, 21))
plt.barh(classes, iou, align='center', color='royalblue')

for index, value in enumerate(iou):
    plt.text(value, index, '{0:.2f}'.format(value), fontsize=16, color='royalblue')

plt.title('Average IOU: {0:.2f}%'.format(np.mean(iou) * 100), fontsize=18)
plt.xlabel('IOU', fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('output/iou.png')
plt.show()
