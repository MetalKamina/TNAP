from PIL import Image
import numpy as np
import os

def clean_data(folder_path,tgt_size):
    dir_list = os.listdir(folder_path)
    arr1 = np.zeros([len(dir_list),tgt_size,tgt_size])
    arr2 = np.zeros([len(dir_list),tgt_size*tgt_size])

    i = 0

    for filename in dir_list:
        f = os.path.join(folder_path, filename)
        im = Image.open(f).convert('L')
        im = np.array(im)

        row_dis = int((im.shape[0]-tgt_size)/2)
        col_dis = int((im.shape[1]-tgt_size)/2)

        im = np.array(im[row_dis:row_dis+tgt_size, col_dis:col_dis+tgt_size])
        arr1[i] = im
        arr2[i] = im.flatten()
        i+=1
    return arr1, arr2