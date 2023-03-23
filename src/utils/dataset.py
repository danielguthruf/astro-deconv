import os
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathing import *
from astropy.visualization import LogStretch,SqrtStretch,AsinhStretch,SinhStretch
from monai.transforms import *
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from utils.custom import PoissonNoise, RemoveZeroImage, ReplaceValue

class AstroDataset:
    def __init__(self, 
                 query_id, 
                 data_split, 
                 pin_memory=True,
                 dataset_type="cache",
                 cache_rate=1.0,
                 num_workers=1,
                 train_validation_ratio=0.8,
                 batch_size=16,
                 crop=None,
                 ):
        self.query_id = query_id
        self.data_split = data_split
        self.pin_memory = pin_memory
        self.dataset_type= dataset_type
        self.processed_data_path = f"{PROCESSED_DATA_PATH}/{query_id}"
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.batch_size=batch_size
        self.train_validation_ratio=train_validation_ratio
        self.data = makeDataDictS(self.processed_data_path,train_validation_ratio,data_split)
        self.transforms = getTransformS(self.data_split,crop)
        

    def initalizeTorchDataset(self,num=-1):
        
        dataset = CacheDataset(
                    data=self.data[0:num],
                    transform=self.transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
        )
        return dataset
    
    def initalizeTorchLoader(self,dataset):
        if self.data_split=="train":
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=list_data_collate,
            )
        else:
            loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers)
        return loader
    
def getTransformS(data_split,crop):
    transforms=getDefaultTransforms(data_split,crop)
    return transforms
def getDefaultTransforms(data_split,crop):
    if data_split=="train":
        transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ReplaceValue(keys=["image"]),
                RandCropByPosNegLabeld(**crop
                ),
                CopyItemsd(keys=["image"],
                            names=["noisy_image"]),
                PoissonNoise(keys=["noisy_image"]),
                # ScaleIntensityRanged(
                #     keys=["image","noisy_image"],
                #     a_min=0,
                #     a_max=100,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),

                # RemoveZeroImage(),
                ToTensord(keys=['image','noisy_image'], dtype=torch.float16),
                CastToTyped(keys=['image','noisy_image'], dtype=torch.float16),
        ])
    elif data_split=="validation":
        transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ReplaceValue(keys=["image"]),
                CopyItemsd(keys=["image"],
                            names=["noisy_image"]),
                PoissonNoise(keys=["noisy_image"]),
                # RemoveZeroImage(),
                # ScaleIntensityRanged(
                #     keys=["image","noisy_image"],
                #     a_min=0,
                #     a_max=100,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                ToTensord(keys=['image','noisy_image'], dtype=torch.float16),
                CastToTyped(keys=['image','noisy_image'], dtype=torch.float16),
        ])
    return transforms
def doTransformS(transformS,imageS):
    if len(imageS[0].shape)==2:
        imageS=[np.expand(image,axis=0) for image in imageS]
    return [transformS(image) for image in imageS]
    
def getExpTime(filename):
    match = re.search(r"\d+\.\d+", filename)
    number = match.group()
    return float(number)
def split_data(data, ratio, data_split):
    n = len(data)
    split_idx = int(n * ratio)
    train_data = data[:split_idx]
    validation_data =  data[split_idx:]
    if data_split=="train":
        return train_data
    else:
        return validation_data

def makeDataDict(imagePath):
    return {"image":imagePath,
            "id":imagePath.strip(".npz"),
            "exp_time":getExpTime(imagePath)
            }
  
def makeDataDictS(data_path,train_validation_ratio,data_split):
    list_of_dicts=[]
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            if ".npz" in file_path:
                list_of_dicts.append(makeDataDict(file_path))
            
    list_of_dicts = split_data(list_of_dicts,train_validation_ratio,data_split)
    return list_of_dicts

def getImage(path):
    data=np.load(path)
    return data['arr_0']

def getImageS(pathS,num=1,transformS=None):
    pathS=check_path(pathS)
    imageS=[]
    for path in pathS[0:num]:
        imageS.append(getImage(path))
    if transformS:
        doTransformS(imageS=imageS,transformS=transformS)
    return imageS
        
def stretchImage(image,mode="log"):
    if mode == "log":
        stretch = LogStretch()
    elif mode == "sqrt":
        stretch = SqrtStretch()
    elif mode == "asinh":
        stretch = AsinhStretch()
    elif mode == "sinh":
        stretch = SinhStretch()
    else:
        raise ValueError("Invalid stretch mode specified")
    stretched_image = stretch(image)
    return stretched_image

def stretchImageS(pathS,mode="log",num=1,transformS=None):
    imageS=getImageS(pathS,num,transformS)
    stretched_imageS=[]
    for image in imageS:
        stretched_imageS.append(stretchImage(image,mode))
    return stretched_imageS
    
def plotImageS(pathS,mode="log",num=1,transformS=None):
    if mode:
        imageS = stretchImageS(pathS,mode,num,transformS)
    else:
        imageS = getImageS(pathS,num,transformS)
    
    n_images = len(imageS)
    if n_images == 1:
        n_cols, n_rows = 1, 1
    elif n_images == 2:
        n_cols, n_rows = 2, 1
    elif n_images == 3:
        n_cols, n_rows = 3, 1
    else:    n_cols = int(math.sqrt(n_images))
    n_rows = int(math.ceil(n_images / n_cols))
    print(n_rows)
    print(n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

    for i, image in enumerate(imageS):
        if n_images == 1:
            ax = axes
        else:
            row, col = divmod(i, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]

        ax.imshow(image, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def check_path(x):
    if isinstance(x, str):
        if os.path.exists(x):
            return [x]
        else:
            raise ValueError(f"{x} does not exist!")
    elif isinstance(x, list):
        paths=[]
        if isinstance(x[0], dict):
            for adict in x:
                paths.append(adict['image'])
        else:
            paths=x
        if not all([os.path.exists(v) for v in paths]):
            raise ValueError("Not all paths are valid!")
        return paths
    else:
        raise ValueError("You did not provide, a path, a dict of paths or a list of paths!")
