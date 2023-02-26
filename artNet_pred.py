from PIL import Image
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import torch
from torchvision import transforms
import numpy as np



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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(device)
    model = model.to(device)
    model.load_state_dict(torch.load("artnet_75_sgd.pth"))
    
    img = Image.open(image_path)
    imgsize = np.array(img).shape
    mean = [0.5, 0.5, 0.5] 
    std = [0.5, 0.5, 0.5]
    transform_norm = transforms.Compose([transforms.CenterCrop(min(imgsize[0],imgsize[1])),transforms.Resize(512),transforms.ToTensor(),transforms.Normalize(mean, std)])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        if(output.cpu().numpy()[0][1] < 0):
            return 1
        return 0