from collections import defaultdict

CIFAR100_LABELS = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
)

CIFAR100_SUPERCLASSES = {
    'aquatic mammals':              ('beaver', 'dolphin', 'otter', 'seal', 'whale'),
    'fish':                         ('aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'),
    'flowers':                      ('orchid', 'poppy', 'rose', 'sunflower', 'tulip'),
    'food containers':              ('bottle', 'bowl', 'can', 'cup', 'plate'),
    'fruit and vegetables':         ('apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'),
    'household electrical devices': ('clock', 'keyboard', 'lamp', 'telephone', 'television'),
    'household furniture':          ('bed', 'chair', 'couch', 'table', 'wardrobe'),
    'insects':                      ('bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'),
    'large carnivores':             ('bear', 'leopard', 'lion', 'tiger', 'wolf'),
    'large man-made outdoor things':('bridge', 'castle', 'house', 'road', 'skyscraper'),
    'large natural outdoor scenes': ('cloud', 'forest', 'mountain', 'plain', 'sea'),
    'large omnivores and herbivores':('camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'),
    'medium-sized mammals':         ('fox', 'porcupine', 'possum', 'raccoon', 'skunk'),
    'non-insect invertebrates':     ('crab', 'lobster', 'snail', 'spider', 'worm'),
    'people':                       ('baby', 'boy', 'girl', 'man', 'woman'),
    'reptiles':                     ('crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'),
    'small mammals':                ('hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'),
    'trees':                        ('maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'),
    'vehicles 1':                   ('bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'),
    'vehicles 2':                   ('lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')
}


CIFAR100_ENCODING = defaultdict(lambda: 100)
CIFAR100_DECODING = defaultdict(lambda: 'unknown')
for i, label in enumerate(CIFAR100_LABELS):
    CIFAR100_ENCODING[label] = i
    CIFAR100_DECODING[i] = label


CIFAR100_SUPERCLASS_ENCODING = defaultdict(lambda: len(CIFAR100_SUPERCLASSES))
CIFAR100_SUPERCLASS_DECODING = defaultdict(lambda: 'unknown')
for i, label in enumerate(CIFAR100_SUPERCLASSES):
    CIFAR100_SUPERCLASS_ENCODING[label] = i
    CIFAR100_SUPERCLASS_DECODING[i] = label


def get_superclass(label):
    for (key, value) in CIFAR100_SUPERCLASSES.items():
        if label in value:
            return key
    raise RuntimeError('label not found: ', label)
