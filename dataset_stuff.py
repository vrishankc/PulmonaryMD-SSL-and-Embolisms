import torch
from torch.utils.data import Dataset
device = "gpu" if torch.cuda.is_available() else "cpu"


import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

import random
from style_transfer import *

from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        # self.targets = torch.LongTensor(targets)\n",
        self.transform = transform
    def __getitem__(self, index):
        
        x = torch.load(str(self.data[index]))
        x = x.unsqueeze_(0)
        x = torch.repeat_interleave(x, 3, dim = 0)
        
        
        if self.transform:
            x = self.transform(x) 
        
        '''
        image_options = {
            0: 'nasa.jpeg',
            1: 'starry.jpeg',
            2: 'stars.jpeg'
        }
        num = random.randrange(0, 3)
        style_transfer_confirmation = {
            0: False,
            1: True
        }
        conf_num = random.randrange(0, 2)

        
        
        if style_transfer_confirmation[conf_num]:
            # set up style image
            style_im = Image.open(f'{image_options[num]}')
            style_im = style_im.resize((224, 224))
            
            style_im = np.asarray(style_im)
            style_im = torch.from_numpy(style_im)
            style_im = torch.transpose(style_im, 0, 2)
            style_im = style_im.unsqueeze_(0)
            
            #style_im = transforms.ToPILImage()(style_im)
            
            # set up content image
            
        
            x = transforms.Resize(224)(x)
            content_im = x
            #content_im, x = np.asarray(content_im), np.asarray(x)
            #content_im, x = torch.from_numpy(content_im), torch.from_numpy(x)
            content_im, x = torch.transpose(content_im, 0, 2), torch.transpose(x, 0, 2)
            
            
            content_im, x = content_im.unsqueeze_(0), x.unsqueeze_(0)
            
            #content_im = transforms.ToPILImage()(content_im)
            x = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_im, style_im, x)
            
        
        else:
            pass
        '''
            
        
        return x

    def __len__(self):
        return len(self.data)
