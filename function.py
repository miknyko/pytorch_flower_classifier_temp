import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim

from torchvision import datasets,transforms,models
import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load',help = 'load the checkpoint or new training',default = '1')
    parser.add_argument('--data_dir',help = 'path to the image folder')
    parser.add_argument('--save_dir',help = 'path to the training checkpoint')
    parser.add_argument('--arch',help = 'the architechture of the network',default = 'vgg')
    parser.add_argument('--lr',help = 'the learning rate',default = 0.001)
    parser.add_argument('--hidden units',help = 'the hidden units',default = 512)
    parser.add_argument('--epochs',help = 'setting the epochs',type = int, default = 20)
    parser.add_argument('--device',help = 'CPU OR CUDA',default = 'cuda')
    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--topk',help = 'print the top N class',default = 3)
    parser.add_argument('--category_names',help = 'the index of the labels to classes',default = 'cat_to_name.json')
    parser.add_argument('--device',help = 'CPU OR CUDA',default = 'cuda')
    parser.add_argument('--arch',help = 'the architechture of the network',default = 'vgg')
    parser.add_argument('--save_dir',help = 'path to the training checkpoint')
    parser.add_argument('--dirpic',help = 'path to the picture to test')
def accuracy_test(model,dataloader):
    correct = 0
    total = 0
    model.cuda()
    with torch.no_grad():
        for data in dataloader:
           
            images,labels = data
            images,labels = images.to('cuda'),labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
           
    
    print('the accuracy is {:.4f}'.format(correct/total))
    
    
#create the deep learning function

def deep_learning (model,trainloader,epochs,print_every,criterion,optimizer,device='cuda'):
    epochs = epochs
    print_every = print_every
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii , (inputs,labels) in enumerate(trainloader):
            steps += 1
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            
            #forward and backward
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                #test the accuracy
               
                
                print('EPOCHS : {}/{}'.format(e+1,epochs),
                      'Loss : {:.4f}'.format(running_loss/print_every))
                accuracy_test(model,validloader)
                
#def checkpoint loading function
def network_loading(model,ckp_path):
    state_dict = torch.load(ckp_path)
    model.load_state_dict(state_dict)
    print('The Network is Loaded')

def network_saving():
    torch.save(fmodel.state_dict(),'ckp')
    
    print('The Network is Saved')
    
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #adjust the size
    pic = Image.open(image)
    if pic.size[0] < pic.size[1]:
        ratio = float(256) / float(pic.size[0])
    else:
        ratio = float(256) / float(pic.size[1])
    
    new_size = (int(pic.size[0]*ratio),int(pic.size[1]*ratio))
    
    pic.thumbnail(new_size)
    
    #crop the 224*224
    
    pic = pic.crop([pic.size[0]/2-112,pic.size[1]/2-112,pic.size[0]/2+112,pic.size[1]/2+112])
    
    #converting
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    np_image = np.array(pic)
    np_image = np_image/255
    
    for i in range(2):
        
        np_image[:,:,i] -= mean[i]
        np_image[:,:,i] /= std[i]
    
    np_image = np_image.transpose((2,0,1))
    np_image = torch.from_numpy(np_image)
    np_image = np_image.float()
    print(np_image.type)
    return np_image

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = img.unsqueeze(0)
    result = model(img.cuda()).topk(topk)
    probs= []
    classes = []
    a = result[0]
    b = result[1].tolist()
    
    for i in a[0]:
        probs.append(torch.exp(i).tolist())
    for n in b[0]:
        classes.append(str(n+1))
    
    return(probs,classes)