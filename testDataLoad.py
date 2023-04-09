print("getting files...")
import glob
import os
import numpy as np
import torch 
from torchvision.transforms import transforms
#from dataLoad import preprocess, hu_transformation, remove_noise, image_window

path = r'testDataFive/batch_five/*.npy'
files = glob.glob(path)
#print(files[38:len(files)])
#print(files[38])
numOfFiles = len(files)
#device = "cuda"
#torch.device("cuda")
for i in range(numOfFiles):
    file = np.load(str(files[i]))
    numOfImages = file.shape[0]
    for j in range(numOfImages):
        image = file[j]
        #image = preprocess(np.int16(image))
        image = torch.Tensor(image)
        torch.save(image, f'testDataFive/batch_five/{i+1}_{j+1}.pt')




