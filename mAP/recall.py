import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

with open('output/output.txt', "r") as f:
    lines = f.readlines()

precision = []
recall = []
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

    if line[0] == 'Recall ':
        line = line[1:][0].strip("[]").split(',')
        new_line = [l.replace("'", "") for l in line]

        for line in new_line:
            if line == '':
                new_line[0] = '0.0'
        line = [float(l) for l in new_line]
        recall.append(np.mean(line))

recall = np.array(recall)
classes = np.array(classes)

arg = np.argsort(recall)
print(arg)
recall = recall[arg]
classes = classes[arg]
plt.figure(figsize=(9, 21))
plt.barh(classes, recall, align='center', color='royalblue')

for index, value in enumerate(recall):
    plt.text(value, index, '{0:.2f}'.format(value), fontsize=16, color='royalblue')

plt.title('Mean Recall: {0:.2f}%'.format(np.mean(recall) * 100), fontsize=18)
plt.xlabel('Average Recall', fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('output/recall.png')
plt.show()
