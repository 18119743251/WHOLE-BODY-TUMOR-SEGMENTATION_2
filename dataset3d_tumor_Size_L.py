import random
import torch
import torch.nn as nn
import h5py
import os
import SimpleITK as sitk
import json
import numpy as np
import nibabel as nib
from tqdm.notebook import tqdm
from scipy.ndimage.interpolation import zoom
from monai.transforms import ScaleIntensityRanged 
from monai.transforms import EnsureChannelFirstd
import lz4.frame
import pickle
from pathlib import Path

class segmentation3dRAMNew_tumor(torch.utils.data.Dataset): #  读取训练集的数据以及标签
    def __init__(self, paths, shape= (128, 128, 128), ct=True, mask=True, norm=True):
        self.paths = paths
        self.shape = shape
        self.ct = ct
        self.mask = mask
        self.norm = norm
        #self.total = '/opt/data/private/pipeline/venv/TOTAL2.0/'
        self.pet_ct = '/opt/data/private/pipeline/venv/PET_CT2.0/'
        self.newseg = '/opt/data/private/pipeline/venv/Newresample666/NewSeg/'

    def __getitem__(self, idx):
        path = self.paths[idx]
        your_path = Path(path)
        your_path = your_path.stem
        
        # path_total = self.total+str(your_path)[0:33]+'-p.nii.gz'+'/'+'RESct.nii'  #这里得到对应的total
        # imgs = sitk.ReadImage(path_total)
        # temp_total = sitk.GetArrayFromImage(imgs)
        
        path_pt = self.pet_ct+str(your_path)[0:33]+'-p'+'/'+'RESsuv.nii.gz'  #这里得到对应的pet
        imgspet = sitk.ReadImage(path_pt)
        temp_pet = sitk.GetArrayFromImage(imgspet)
        
        path_ct = self.pet_ct+str(your_path)[0:33]+'-p'+'/'+'RESctres.nii.gz'  #这里得到对应的ct
        imgsct = sitk.ReadImage(path_ct)
        temp_ct = sitk.GetArrayFromImage(imgsct)
        
        path_seg = self.newseg+str(your_path)[0:33]+'-p'+'/'+'NewSeg.nii.gz'  #这里得到对应的seg
        imgsseg = sitk.ReadImage(path_seg)
        temp_seg = sitk.GetArrayFromImage(imgsseg)
        
        #print(path_total, path_seg)
        #print(path,your_path)
        self.gt = []
        self.pt = []
        self.data = []
        self.data2 = []
        #f = h5py.File(path, 'r')
        temp_pt = temp_pet
        temp_gt = temp_seg
        temp_ct = temp_ct # temp_pt,ct,gt此处都是数组形式
        #f.close()

        #img_array = np.asarray(temp_ct)
        #out = sitk.GetImageFromArray(img_array)
        #writepath = '/opt/data/private/pipeline/venv/ct/test/' +str(your_path)+'1'+'.nii.gz' # 保存图像路径
        #sitk.WriteImage(out, writepath)

        #这里需要得到对应的label，这个label是total segmentator处理后的数据，相当于pre-train
        pt = temp_pt
        ct = temp_ct
        gt = temp_gt
        scale = ScaleIntensityRanged(keys='image', a_min=-800.0, a_max=400.0, b_min=0.0, b_max=1.0, clip=True)

        if self.norm:
            #ct = (ct - np.min(ct)) / np.percentile(ct, 99)
            pt = (pt - np.min(pt)) / np.percentile(pt, 99)
            
        temp_in3 = [ct]
        self.data2.append(np.array(temp_in3))
        ct = self.data2[0]
        ctt = [{'image':ct,'label':pt}]
        
        ct2 = scale(ctt[0])
        ct = np.array(ct2['image'])
        #print('the shape of ct2 is:',ct2.shape)
        depth = pt.size(0)
        high = pt.size(1)
        wide = pt.size(2)


        d = random.randrange(0, depth-127)
        h = random.randrange(0, high-127)
        w = random.randrange(0, wide-127)

        pt = pt[d:d + 128, h:h + 128, w:w + 128]
        ct = ct[:, d:d + 128, h:h + 128, w:w + 128]
        gt = gt[d:d + 128, h:h + 128, w:w + 128]
        #tolt2 = tolt[hh:hh+64, :, :]
        #gt2 = gt[hh:hh + 64, :, :]
        #pt2 = pt[hh:hh + 64, :, :]
        #ct2 = ct2[:,hh:hh + 64, :, :]

        #y = torch.from_numpy(gt2)[None] #加上[none]就是（1，128，128，128），不加就是（128，128，128）
        
        #f.close()
        #print(x2.shape,x4.shape)
        pt = torch.from_numpy(pt)[None]
        ct = torch.from_numpy(ct)
        gt = torch.from_numpy(gt)[None]
        
        return pt.float(), ct.float(), gt
        #顺序是pt,ct,gt(label)
        

    def __len__(self):
        return len(self.paths)
