
import torch
has_gpu = torch.cuda.is_available()

has_mps = getattr(torch,'has_mps',False)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device("cuda")


from transformations import Transformations
from dataset_stuff import MyDataset


import torch
from skimage import transform
import numpy as np
import pydicom 
from torchvision import transforms
import SimpleITK as sitk
import scipy.ndimage as ndimage
from skimage import morphology

# PREPROCESSING UNIT

# redo code for copyright

print("getting files...")
import glob
import numpy as np
import random

path = r'testDataFive/batch_five/*.pt'
files = glob.glob(path)
files = random.sample(files, len(files))
numOfFiles = len(files)

#arr = np.empty((numOfFiles, totalNumOfImages, 512, 512))

def hu_transformation(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = slope * image + intercept
    return hu_image

def image_window(image, center, width):
    img_min = center - width // 2
    img_max = center + width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def remove_noise(windowed_image, display = False):
    segmentation = morphology.dilation(windowed_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0
    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    masked_image = mask * windowed_image

    return masked_image

def preprocess(image):
    img = sitk.GetImageFromArray(image)
    sitk.WriteImage(img, "name.dcm")
    medical_image = pydicom.read_file("name.dcm", force = True)
    
    hu_image = hu_transformation(medical_image, image)
    windowed_image = image_window(hu_image, 400, 3000)
    
    final = remove_noise(windowed_image)
    #minimum = abs(final.min())
    #maximum = final.max()
    #final += abs(final.min())
    #final *= 1 /(minimum + final.max())

    return final

print("time to apply preprocessing")
'''from torchvision.transforms import transforms
transformation = transforms.CenterCrop(450)
for i in files:
    arr = torch.load(str(i)).numpy()
    arr = preprocess(np.int16(arr))
    arr = torch.Tensor(arr)
    minimum = abs(arr.min())
    maximum = arr.max()
    arr += minimum
    arr *= 1 /(minimum + maximum)
    arr = transformation(arr)
    torch.save(arr, str(i))'''

# AUGMENTATION UNIT
print("augmentation unit built...")
transform = transforms.Compose([
    transforms.ColorJitter(brightness = 1.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=224),
    transforms.RandomRotation(45),
])
'''transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=96),
    transforms.RandomRotation(45),
    transforms.Normalize((0.5,), (0.5,))
])'''
print("trainDataset and valDataset being set up")
train_split_num = int(0.9 * int(numOfFiles))


trainDataset = MyDataset(files[0:train_split_num], transform=Transformations(transform, numOfViews=2))
#train_dataloader = DataLoader(trainDataset, batch_size = 64)

valDataset = MyDataset(files[train_split_num:numOfFiles], transform=Transformations(transform, numOfViews=2))
#val_dataloader = DataLoader(valDataset, batch_size = 64)
print("Data Successfully Loaded!")
