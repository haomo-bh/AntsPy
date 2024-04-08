import numpy as np
import os 
import ants
import glob


def Mse(x, y):
    l1 = x - y
    return np.sqrt(np.sum(l1 * l1))

def Tre(ct_p, mr_p, nums=9):
    dst = np.zeros(nums)
    cnt = 0
    for i in range(1, nums+1):
        point1 = np.where(ct_p==i)
        point2 = np.where(mr_p==i)
        if len(point1[0]) == 1 and len(point2[0]) == 1:
            cnt += 1
            point1_new = np.array([point1[0].item(), point1[1].item(), point1[2].item()])
            point2_new = np.array([point2[0].item(), point2[1].item(), point2[2].item()])
            dst[i-1] = Mse(point1_new, point2_new)
            # print(f"第{i}个点的距离为{dst[i-1]}", file=f)
        else:
            # 表示此时距离不存在
            dst[i-1] = -1
    return cnt, np.round(dst, 2)
        

def Get_Mask_Points(mask, file_name, mode, nums=9):
    with open(file_name, mode=mode) as f2:
        for i in range(1, nums+1):
            points = np.where(mask==i)
            if len(points[0]) == 1:
                f2.write(str(points[0].item()))
                f2.write(' ')
                f2.write(str(points[1].item()))
                f2.write(' ')
                f2.write(str(points[2].item()))
            else:
                f2.write("-1 -1 -1")
            f2.write('\n')


# 数据存储的路径
img_path = r"D:/Transmorph_Frame/Data/seg_need_nii/raw_images/new_nii_resampled_normalized_files/new_nii_resampled_normalized_cropped_files/"
point_path = r"D:/Transmorph_Frame/Data/seg_need_nii/raw_tumors_seg/new_nii_resampled_normalized_cropped_files/"
file_names = os.listdir(img_path)
CT_imgs = glob.glob(img_path + '*CT*.nii.gz')
MR_imgs = glob.glob(img_path + '*MR*.nii.gz')
CT_points = glob.glob(point_path + '*CT*.nii.gz')
MR_points = glob.glob(point_path + '*MR*.nii.gz')
# 将需要的文件进行打包
test_zipped_imgs = zip(CT_imgs, MR_imgs, CT_points, MR_points)
files_cnt = 0
tre_arr = []
tre_arr_new = []
# 图片读取
with open('./dst.txt', "w") as f:
    with open('./dst_new.txt', "w") as f1:
        for fixed, moving, fixed_points, moving_points in test_zipped_imgs:
            mode = 'a'
            if files_cnt==0:
                mode='w'
            f_img = ants.image_read(fixed)
            m_img = ants.image_read(moving)
            f_points = ants.image_read(fixed_points)
            m_points = ants.image_read(moving_points)
            # 图像配准
            mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Affine', reg_iterations=(160, 80, 40))
            warped_points = ants.apply_transforms(fixed=f_points, moving=m_points, transformlist=mytx['fwdtransforms'],
                                                interpolator='nearestNeighbor')
            # print("配准前的点位距离如下：", file=f)
            # print("*****************************", file=f)
            cnt, dst = Tre(f_points.numpy(), m_points.numpy(), 9)
            Get_Mask_Points(f_points.numpy(), file_name='./ct_points.txt', mode=mode)
            Get_Mask_Points(m_points.numpy(), file_name='./mr_points.txt', mode=mode)
            Get_Mask_Points(warped_points.numpy(), file_name='./mr_points_new.txt', mode=mode)
            # 用于返回距离的txt
            # for i in range(0, 9):
            #     f.write(str(dst[i].item()))
            #     if i==8:
            #         f.write('\n')
            #     else:
            #         f.write(' ')
            # print("配准后的点位距离如下：", file=f)
            # print("*****************************", file=f)
            cnt_new, dst_new = Tre(f_points.numpy(), warped_points.numpy(), 9)
            # # 用于返回距离的txt
            # for i in range(0, 9):
            #     f1.write(str(dst_new[i].item()))
            #     if i==8:
            #         f1.write('\n')
            #     else:
            #         f1.write(' ')
            # print(f'第{i}组{file_names[(i-1)*2][0:10]}的配准前Tre为{mean}, 可用点数为{cnt}', file=f)
            # print(f'第{i}组{file_names[(i-1)*2][0:10]}配准后的Tre为{mean_new}, 可用点数为{cnt_new}', file=f)
            # tre_arr.append(mean.item())
            # tre_arr_new.append(mean_new.item())
            files_cnt += 1
            # print('***********************************************', file=f)
            # print(f'配准前平均Tre为{np.mean(np.array(tre_arr))}', file=f)
            # print(f'配准后平均Tre为{np.mean(np.array(tre_arr_new))}', file=f)
