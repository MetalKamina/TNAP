from PIL import Image
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import torch
from torchvision import transforms
import numpy as np
import pickle
import os
from PIL import Image
import numpy as np

#net declaration
class Net(nn.Module):
    def __init__(self,device):
        self.device = device
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=4,kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        self.conv2 = Conv2d(in_channels=4, out_channels=8,kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        self.conv3 = Conv2d(in_channels=8, out_channels=16,kernel_size=(5, 5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        self.conv4 = Conv2d(in_channels=16, out_channels=32,kernel_size=(5, 5))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        self.l1 = nn.Linear(800, 64)
        self.l2 = nn.Linear(64, 2)
        self.flat = nn.Flatten()
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        out = self.flat(x)
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out
 
    def training_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, pred = torch.max(out, 1)
        accuracy = torch.tensor(torch.sum(pred==labels).item()/len(pred))
        return [loss.detach(), accuracy.detach()]
 

def artnet_predict(image_path):
    #set up/load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(device)
    model = model.to(device)
    model.load_state_dict(torch.load("artnet.pth"))
    
    #open and transform/normalize image
    img = Image.open(image_path)
    imgsize = np.array(img).shape
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.CenterCrop(min(imgsize[0],imgsize[1])),transforms.Resize(512),transforms.ToTensor(),transforms.Normalize(mean, std)])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    
    #evaluate image, returning 1 if the image is AI and 0 if it's manmade
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        if(output.cpu().numpy()[0][1] < 0):
            return 1
        return 0

def predict_image(file_path):
    imgpath = os.path.join(file_path)
    im = Image.open(imgpath).convert('L')
    
    #load forest classifier
    f = open('rforest.sav', 'rb')
    model = pickle.load(f)
    f.close()

    f = open('kneighbors.sav', 'rb')
    neighbors = pickle.load(f)
    f.close()

    #clean image and prepare for prediction
    imt = np.array(im)
    if((imt.shape[0] >= 512 and imt.shape[1] >= 512)):
        im = np.array(im)
        if(im.shape[0] >= 512*2 and im.shape[1] >= 512*2):
            im1 = np.array(im[im.shape[0]-512-512:im.shape[0]-512, im.shape[1]-512-512:im.shape[1]-512]).flatten()
            im2 = np.array(im[im.shape[0]-512-512:im.shape[0]-512, im.shape[1]-512:]).flatten()
            im3 = np.array(im[im.shape[0]-512:,im.shape[1]-512-512:im.shape[1]-512]).flatten()
            im4 = np.array(im[im.shape[0]-512:,im.shape[1]-512:]).flatten()

            tmp = model.predict([im1,im2,im3,im4])
            if(np.sum(tmp) == 2):
                rforest_pred = int(tmp[3])
            elif(np.sum(tmp) >= 3):
                rforest_pred = 1
            else:
                rforest_pred = 0

            
            
        else:
            im = np.array(im[im.shape[0]-512:,im.shape[1]-512:]).flatten()
            rforest_pred = model.predict([im])[0]
    else:
        im.resize(512,512)
        im = np.array(im)
        im = np.array(im[im.shape[0]-512:,im.shape[1]-512:]).flatten()
        rforest_pred = model.predict([im])[0]

    nn_pred = artnet_predict(file_path)

    if(rforest_pred != nn_pred):
        if((imt.shape[0] >= 512 and imt.shape[1] >= 512)):
            im = np.array(im)
            if(imt.shape[0] >= 512*2 and imt.shape[1] >= 512*2):
                tmp = neighbors.predict([im1,im2,im3,im4])
                if(np.sum(tmp) == 2):
                    neighbor_pred = int(tmp[3])
                elif(np.sum(tmp) >= 3):
                    neighbor_pred = 1
                else:
                    neighbor_pred = 0
            else:
                neighbor_pred = model.predict([im])[0]
        else:
            im.resize(512,512)
            im = np.array(im)
            im = np.array(im[im.shape[0]-512:,im.shape[1]-512:]).flatten()
    else:
        neighbor_pred = rforest_pred


    #return predictions
    if((rforest_pred == 1) and (nn_pred == 1) and (neighbor_pred == 1)):
        return ("AI","🙂")
    elif((rforest_pred == 0) and (nn_pred == 1)):
        return ("Manmade","🤨")
    elif((rforest_pred == 1) and (nn_pred == 0) and (neighbor_pred == 1)):
        return ("AI","🤨")
    elif((rforest_pred == 1) and (nn_pred == 1) and (neighbor_pred == 0)):
        return ("AI","🤨")
    elif((rforest_pred == 0) and (nn_pred == 0) and (neighbor_pred == 1)):
        return ("Manmade","🤨")
    elif((rforest_pred == 0) and (nn_pred == 1) and (neighbor_pred == 0)):
        return ("Manmade","🤨")
    elif((rforest_pred == 1) and (nn_pred == 0) and (neighbor_pred == 0)):
        return ("Manmade","🤨")
    else:
        return ("Manmade","🙂")
