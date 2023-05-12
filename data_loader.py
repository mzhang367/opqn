import torch
import torchvision.transforms as transforms
from torchvision import datasets

import os

def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
    to_tensor = transforms.ToTensor()
    if dataset!="vggface2":
        trainPaths = os.path.join(data_dir, dataset, "train") 
        testPaths = os.path.join(data_dir, dataset, "test")
    else:
        if cross_eval: # vgggface2 cross-dataset retrieval uses another train-test splits from standard retrieval
            trainPaths = os.path.join(data_dir, "vggface2", "cross_train") 
            testPaths = os.path.join(data_dir, "vggface2", "cross_test")
        else:
            trainPaths = os.path.join(data_dir, "vggface2", "train") 
            testPaths = os.path.join(data_dir, "vggface2", "test")
    trainset = datasets.ImageFolder(root=trainPaths, transform=to_tensor)
    testset = datasets.ImageFolder(root=testPaths, transform=to_tensor)
    if cross_eval:
        transform_train = torch.nn.Sequential(
                    transforms.Resize(120),
                    transforms.CenterCrop(112),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

        transform_test = transform_train

    else:
        if datasets=="vggface2":
            transform_train = torch.nn.Sequential(
                    transforms.Resize(120),
                    transforms.RandomCrop(112),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )

            transform_test = torch.nn.Sequential(
                    transforms.Resize(120),
                    transforms.CenterCrop(112),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
        
        else:
            transform_train = torch.nn.Sequential(
                    transforms.Resize(35), 
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
            )

            transform_test = torch.nn.Sequential(
                    transforms.Resize(35), 
                    transforms.CenterCrop(32),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
            )
    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}
    

    

