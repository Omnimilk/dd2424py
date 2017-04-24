import urllib.request
import os
import pickle
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

class CIFAR10:

    def __init__(self):
        self.download_dataset()
        self.load_labels()
        self.loaded_batches = {}

    def load_labels(self):
        with open('cifar-10-batches-py/batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.labels = [x.decode('ascii') for x in data[b'label_names']]

    def download_dataset(self):
        if not os.path.isdir('cifar-10-batches-py'):
            file, _ = urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "temp.tar.gz")
            with tarfile.open(file, "r:gz") as tar:
                tar.extractall()
            os.remove(file)

    def get_batch(self, batch_name):
        if not batch_name in self.loaded_batches:
            with open('cifar-10-batches-py/' + batch_name, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            self.loaded_batches[batch_name] = {
                'batch_name': data[b'batch_label'],
                'images': np.divide(data[b'data'], 255),
                'labels': data[b'labels']
            }
        return self.loaded_batches[batch_name]

    def get_batches(self, *args):
        batches = [self.get_batch(name) for name in args]
        return {
            'batch_name': ", ".join(args),
            'images': np.vstack([b['images'] for b in batches]),
            'labels': reduce(lambda acc, v: acc + v['labels'], batches, [])
        }

def show_image(img, label='', interpolation='gaussian'):
    squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
    plt.imshow(squared_image, interpolation=interpolation)
    plt.axis('off')
    plt.title(label)
    plt.show()

cifar = CIFAR10()
training = cifar.get_batches('data_batch_1', 'data_batch_2')
test = cifar.get_batches('data_batch_3')
validation = cifar.get_batches('test_batch')

for i in range(10):
    show_image(training['images'][i], cifar.labels[training['labels'][i]])
