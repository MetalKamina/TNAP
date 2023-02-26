import pickle
import os
from PIL import Image
import numpy as np
from artNet_pred import artnet_predict

def predict_image(file_path):
    imgpath = os.path.join(file_path)
    im = Image.open(imgpath).convert('L')
    im = np.array(im)

    f = open('rforest76.sav', 'rb')
    model = pickle.load(f)
    f.close()

    if((im.shape[0] >= 512 and im.shape[1] >= 512)):
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
    nn_pred = artnet_predict(file_path)

    if(rforest_pred and nn_pred):
        return (1,1)
    elif((not rforest_pred) and nn_pred):
        return (1,0)
    elif(rforest_pred and (not nn_pred)):
        return (0,1)
    else:
        return (0,0)
