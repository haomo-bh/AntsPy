import ants
import matplotlib.pyplot as plt
import time
import numpy as np
import nibabel as nib
import glob
import os
from scipy.ndimage.interpolation import zoom


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def mk_grid_img(grid_step, line_thickness=1):
    grid_img = np.zeros((224 , 192, 112))
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i+line_thickness-1, :, :] = 1
    return grid_img

def flow_as_rgb(flow, slice_num):
    flow = flow[:, :, :, slice_num]
    flow_rgb = np.zeros((flow.shape[1], flow.shape[2], 3))
    for c in range(3):
        flow_rgb[..., c] = flow[c, :, :]
    lower = np.percentile(flow_rgb, 2)
    upper = np.percentile(flow_rgb, 98)
    flow_rgb[flow_rgb < lower] = lower
    flow_rgb[flow_rgb > upper] = upper
    flow_rgb = (((flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min())))
    plt.figure()
    plt.imshow(flow_rgb, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    print(lower)
    print(upper)


# x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
# y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
# x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
# y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
train_path = r'D:/Transmorph_Frame/Data/training_validation_datas_5/train_nii/'
train_tumor = r'D:/Transmorph_Frame/Data/training_validation_datas_5/train_seg_nii/'
train_ct_imgs = glob.glob(train_path + "*CT*.nii.gz")
train_mr_imgs = glob.glob(train_path + "*MR*.nii.gz")
train_ct_tumors = glob.glob(train_tumor + "*CT*.nii.gz")
train_mr_tumors = glob.glob(train_tumor + "*MR*.nii.gz")
x = ants.image_read(train_mr_imgs[0])
y = ants.image_read(train_ct_imgs[0])
x_ants = ants.image_read(train_mr_tumors[0])
y_ants = ants.image_read(train_ct_tumors[0])
slice_num = 52

# x_ants = ants.from_numpy(x_seg.astype(np.float32))
# y_ants = ants.from_numpy(y_seg.astype(np.float32))
start = time.time()
# reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
transform_list = ['./test.nii.gz', './test.mat']
def_seg = ants.apply_transforms(fixed=y_ants,
                 moving=x_ants,
                #  transformlist=reg12['fwdtransforms'],
                transformlist=transform_list,
                 interpolator='nearestNeighbor',)
end = time.time()
print(end - start)
def_out = ants.apply_transforms(fixed=y,
                 moving=x,
                #  transformlist=reg12['fwdtransforms'],
                 transformlist=transform_list,
                 )

# flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
flow = np.array(nib_load(transform_list[0]), dtype='float32', order='C')
flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)

grid_img = mk_grid_img(8, 1)
grid_img = ants.from_numpy(grid_img.astype(np.float32))
def_grid = ants.apply_transforms(fixed=y,
                moving=grid_img,
                #   transformlist=reg12['fwdtransforms'], 
                transformlist=transform_list,
                )
defout = def_grid.numpy()[:,:, slice_num]
plt.figure()
plt.imshow(defout, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.show()

flow_as_rgb(flow, slice_num)

