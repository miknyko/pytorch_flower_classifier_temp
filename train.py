# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import argparse
import function

from collections import OrderedDict

def main():
    
    in_arg = get_input_args()
#Input database dir
    train_dir = in_arg.data_dir+'/train'
    valid_dir = in_arg.data_dir+'/valid'
    test_dir = in_arg.data_dir+'/test'

# TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485,0.456,0.406),
                                                                (0.229,0.224,0.225))])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                     [0.229,0.224,0.225])])



    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform = test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform = test_valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = 32)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size = 20)

    
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

    # TODO: Build and train your network
    fmodel = models[in_arg.arch]
    #Freeze parameters in features
    for param in fmodel.parameters():
        param.require_grad = False

    #create new classifier and replace the old one


    classifier = nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(25088,4096)),
                                 ('relu1',nn.ReLU()),
                                 ('fc2',nn.Linear(4096,1000)),
                                 ('relu2',nn.ReLU()),
                                 ('fc3',nn.Linear(1000,102)),
                                 ('output',nn.LogSoftmax(dim=1))
    ]))
    fmodel.classifier = classifier
    #LOAD THE CHECKPOINT OR NOT
    if in_arg.load == 1:

        network_loading(fmodel,in_arg.save_dir)
    #customize the criterion and the optimizer

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(fmodel.classifier.parameters(),lr=0.001)

    deep_learning(fmodel,trainloader,in_arg.epochs,40,criterion,optimizer,in_arg.device)
    
    accuracy_test(fmodel,testloader)
    
    network_saving()
    
if __name__ == "__main__":
    main()