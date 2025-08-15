
def imshow(dataset):
    import numpy as np

    fig = plt.figure()
    for i in range(25):
        idx = random.randint(0, len(dataset))
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.grid(False)
        plt.imshow(dataset[idx][0].squeeze(),
                   cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(dataset[idx][1]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_label_distribution(dataset):
    import numpy as np
    import pandas as pd

    labels = np.array(dataset.targets)
    pd.DataFrame(labels).hist()


def print_label_distribution(dataset):
    import numpy as np

    labels = np.array(dataset.targets)
    result = Counter(labels)
    print(result)


def plot_coco_image(img, targets, file_name='exmaple.png', mapping=None):
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from collections import Counter
    import torch
    import torchvision
    import cv2 

    if torch.is_tensor(targets):
        targets = targets.numpy()

    plt.figure(figsize = (100, 100))
    for tg in targets:
        id_ = int(tg[-1])
        bbox = tg[:-1]
        x1, y1, x2, y2 = bbox
        color = (255,255,0)
        fp = open('/home/lb4653/mixture-of-experts-thesis/data/coco/coco.names', "r")
        coco_names = fp.read().split("\n")[:-1]
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2), int(y2)),color, 2)
        if mapping:
            id_ = mapping[id_]
        else:
            id_ = str(id_)
        img = cv2.putText(img, str(id_), (int(x1), int(y1)-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color, thickness=1)
    plt.imshow(np.array(img))
    plt.savefig(file_name)

def rescale_bbox(bb, W, H):
    x, y, w, h = bb
    return [x*W, y*H, w*W, h*H]
