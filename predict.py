# Imports here


import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import argparse
import function
import json

def main():
    in_arg = get_input_args_predict()
    
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
    
    network_loading(fmodel,in_arg.save_dir)
    
    result = predict(in_arg.dirpic,fmodel,in_arg.topk)
    probs = result[0]
    classes = result[1]
    

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    names = []
    for i in classes:
    
        names.append(cat_to_name[i])
    
    print('the top possible flower is :')
    for i in names:
        print(i)
    print('with the possiblity of :')
    for i in probs:
        print(i)
    
if __name__ = "__main__":
    main()
    
    
    
    