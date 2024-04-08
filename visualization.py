import numpy as np
import SimpleITK as sitk
import glob
import os

img_path = r"D:\Transmorph_Frame\Data\seg_need_nii\raw_images\new_nii_resampled_normalized_files\new_nii_resampled_normalized_cropped_files/"
imgs = glob.glob(img_path + "*.nii.gz")
for img in imgs:
    image = sitk.ReadImage(img, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array * 4000.0
    new_image = sitk.GetImageFromArray(image_array)
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetDirection(image.GetDirection())
    sitk.WriteImage(new_image, img)

print('end')

    
   