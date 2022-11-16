import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import joblib
'''CFW-60 dataset attributes:
    male, female, asian, white, black, india, youth
    mid, aged, senior, no glasses, eyeglasses
    sunglasses, positive, neural exp. '''


def loading_cfw(train=True):
    """
    CFW60k, for loading images from np array at one time
    :param train:
    :return: tuple of (N*H*W*C, labels)
    """

    if train:
        image_array_ab = joblib.load("./data/cfw_60k/train_ab_array.pkl")
        train_labels_ab = np.load("./data/cfw_60k/train_labels_ab.npy")
        image_array_clf = joblib.load("./data/cfw_60k/train_clf_array.pkl")
        train_labels_clf = np.load("./data/cfw_60k/train_cls_labels.npy")
        train_labels_full = np.concatenate((train_labels_clf, train_labels_ab))
        labels = torch.LongTensor(torch.from_numpy(train_labels_full)).squeeze()
        image_array = np.concatenate((image_array_clf, image_array_ab), axis=0)
        image_array = np.transpose(image_array, (0, 3, 1, 2))
        image_array_torch = torch.from_numpy(image_array)

    else:
        image_array = joblib.load("./data/cfw_60k/test_both_array.pkl")
        image_array = np.transpose(np.array(image_array), (0, 3, 1, 2))

        image_array_torch = torch.from_numpy(image_array)
        test_labels_ab = np.load("./data/cfw_60k/test_labels_ab.npy")
        labels = torch.LongTensor(torch.from_numpy(test_labels_ab)).squeeze()

    return image_array_torch, labels


class MyDataset_transform(Dataset):

    """
    for cfw faces
    """
    def __init__(self, transform, train):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train = train

        self.x, self.y = loading_cfw(train=self.train)
        self.classes = list(np.unique(self.y))

    def __getitem__(self, index):

        imgs, labels = self.x[index], self.y[index]

        if self.transform is not None:
            imgs = self.transform(imgs)
        # labels = torch.from_numpy(labels)
        return imgs, labels

    def __len__(self):
        return len(self.x)


def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''

    mean = torch.zeros(3).cuda()
    std = torch.zeros(3).cuda()
    print('==> Computing mean and std..')
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        for j in range(3):
            mean[j] += inputs[:,j,:,:].mean()
            std[j] += inputs[:,j,:,:].std()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    print(mean, std)


if __name__ == "__main__":

    dataset = MyDataset_transform(transform=None, train=True)
    print(dataset.x.shape)
    print(dataset.y.shape)














