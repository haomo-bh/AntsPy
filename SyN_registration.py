import os
import glob
import ants
import numpy as np
import SimpleITK as sitk


img_path = r"G:/PythonProject/TransMorph/pretrain/Data/SyN(1)/test/"

CT_imgs = glob.glob(img_path + "*CT_resampled_normalized_cropped.nii.gz")
MRI_imgs = glob.glob(img_path + "*MR_resampled_normalized_cropped.nii.gz")
zipped_imgs = zip(CT_imgs, MRI_imgs)
# ants图片的读取
for fixed, moving in zipped_imgs:
    f_img = ants.image_read(fixed)
    m_img = ants.image_read(moving)
    '''
    ants.registration()函数的返回值是一个字典：
        warpedmovout: 配准到fixed图像后的moving图像 
        warpedfixout: 配准到moving图像后的fixed图像 
        fwdtransforms: 从moving到fixed的形变场 
        invtransforms: 从fixed到moving的形变场

    type_of_transform参数的取值可以为:
        Rigid:刚体
        QuickRigid
        Affine:仿射配准，即刚体+缩放
        AffineFast
        ElasticSyN:仿射配准+可变形配准,以MI为优化准则,以elastic为正则项
        SyN:仿射配准+可变形配准,以MI为优化准则
        SyNCC:仿射配准+可变形配准,以CC为优化准则
    '''
    # 图像配准
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN', reg_iterations=(160, 80, 40))
    # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                    interpolator="linear")


    # 将配准后图像的direction/origin/spacing和原图保持一致
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)


    img_name = moving.replace("MR", 'MR_warped')
    # 图像的保存
    ants.image_write(warped_img, img_name)
    print('running')


print('end')
# # 将antsimage转化为numpy数组
# warped_img_arr = warped_img.numpy(single_components=False)
# # 从numpy数组得到antsimage
# img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False)
# # 生成图像的雅克比行列式
# jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False, geom=False)
# ants.image_write(jac, "./result/jac.nii.gz")
# # 生成带网格的moving图像，实测效果不好

# m_grid = ants.create_warped_grid(m_img)
# m_grid = ants.create_warped_grid(m_grid, grid_directions=(False, False), transform=mytx['fwdtransforms'],
#                                  fixed_reference_image=f_img)
# ants.image_write(m_grid, "./result/m_grid.nii.gz")

