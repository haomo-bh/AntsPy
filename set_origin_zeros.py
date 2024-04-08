import SimpleITK as sitk
import glob
import os
import numpy as np

img_path = r'D:/Transmorph_Frame/Data/our_dataset/train_nii/'
print(img_path)
imgs_list = glob.glob(img_path + '*')
i = 0
for img_name in imgs_list:
    img = sitk.ReadImage(img_name)
    img.SetOrigin((0, 0, 0))
    sitk.WriteImage(img, img_name)
    i+=1

print(i)