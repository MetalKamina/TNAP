from PIL import Image
import numpy as np
import os

def clean_data(folder_path,tgt_size,num_items):
    dir_list = os.listdir(folder_path)
    arr1 = []
    arr2 = []

    i = 0
    for filename in dir_list:
        f = os.path.join(folder_path, filename)
        im = Image.open(f).convert('L')
        im = np.array(im)
        
        if(im.shape[0] == 512 and im.shape[1] == 512):
            arr1.append(im)
            arr2.append(im.flatten())
            i+=1
        elif(im.shape[0] >= 512*2 and im.shape[1] >= 512*2):
            im1 = np.array(im[0:512, im.shape[1]-512:])
            arr1.append(im1)
            arr2.append(im1.flatten())

            im2 = np.array(im[0:512, im.shape[1]-512-512:im.shape[1]-512])
            arr1.append(im2)
            arr2.append(im2.flatten())

            im3 = np.array(im[im.shape[0]-512:, im.shape[1]-512:])
            arr1.append(im3)
            arr2.append(im3.flatten())

            im4 = np.array(im[im.shape[0]-512:, im.shape[1]-512-512:im.shape[1]-512])
            arr1.append(im4)
            arr2.append(im4.flatten())
            
            i+=4
        
        if(i >= num_items):
            break
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
            
    return arr2, arr1
