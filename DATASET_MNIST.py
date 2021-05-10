import torch
import scipy.io as sio
import numpy as np


class TRAINSET(object):
    def __init__(self, filename):
        self.mnist = sio.loadmat(filename)

    def __len__(self):
        return self.mnist['mnist_train_images'].shape[1]

    def __getitem__(self, item):
        image = self.mnist['mnist_train_images'][:, item]
        label = self.mnist['mnist_train_labels'][item, 0]
        sample = {'image': image, 'label': label}
        return sample


class TESTSET(object):
    def __init__(self, filename):
        self.mnist = sio.loadmat(filename)

    def __len__(self):
        return self.mnist['mnist_test_images'].shape[1]

    def __getitem__(self, item):
        image = self.mnist['mnist_test_images'][:, item]
        label = self.mnist['mnist_test_labels'][item, 0]
        sample = {'image': image, 'label': label}
        return sample


class DATASET_LOADER(object):
    def __init__(self, dataset, minibatch_size=100):
        self.dataset = dataset
        self.minibatch_size = minibatch_size  # minibatch的大小
        self.minibatch_num = len(self.dataset) // self.minibatch_size  # minibatch的个数

    def __len__(self):
        return self.minibatch_num

    def __getitem__(self, item):
        start, finish = self.minibatch_size * item, self.minibatch_size * (item + 1)
        minibatch = []
        for i in range(start, finish):
            minibatch.append(self.dataset[i])
        minibatch_images = [x['image'] for x in minibatch]
        minibatch_labels = [x['label'] for x in minibatch]
        minibatch_images = np.asarray(minibatch_images)
        minibatch_labels = np.asarray(minibatch_labels)
        minibatch_images = minibatch_images.transpose().reshape(28, 28, self.minibatch_size).transpose((2, 0, 1)) / 255.0
        minibatch_images = torch.FloatTensor(minibatch_images)
        minibatch_images = torch.stack((minibatch_images,), dim=1)
        minibatch_images = torch.nn.functional.pad(minibatch_images, (2, 2, 2, 2), mode='constant', value=0.1)
        minibatch_labels = torch.LongTensor(minibatch_labels)
        return {"image": minibatch_images, "label": minibatch_labels}


if __name__ == "__main__":
    trainset = TRAINSET("./mnist/mnist.mat")
    loader = DATASET_LOADER(trainset, minibatch_size=3)
    images = loader[0]
    print(123)
