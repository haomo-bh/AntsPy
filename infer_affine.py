import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import pickle

def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def main():
    test_dir = r'G:/PythonProject/TransMorph/pretrain/Data' \
        '/zhejiangdaxue/test/test/'
    model_idx = -1
    weights = [1, 0.02]
    model_folder = 'TransMorph_mi_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments_12_20/' + model_folder
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((160, 192, 224), 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    files = glob.glob(test_dir + '*.pkl')
    test_set = datasets.JHUBrainDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    idx = 0
    for data in test_loader:
        print(files[idx].split('\\')[-1].split('.')[0])
        file_name = files[idx].split('\\')[-1].split('.')[0]
        model.eval()
        data = [t.cuda() for t in data]

        ####################
        # Affine transform
        ####################
        x = data[0]  # x_seg = data[2]
        y = data[1]  # y_seg = data[3]
        x_in = torch.cat((x, y), dim=1)
        ct_aff, flow = model(x_in)
        # phan = y.detach().clone(); phan_seg = y_seg
        # phan_seg = nn.functional.one_hot(phan_seg.long(), num_classes=16)
        # phan_seg = torch.squeeze(phan_seg, 1)
        # phan_seg = phan_seg.permute(0, 4, 1, 2, 3).contiguous()
        # ct_tar_seg = AffInferNN(x_seg.float(), mats.float())
        ct_aff = ct_aff.cpu().detach().numpy()[0, 0, :, :, :]
        # ct_tar_seg = ct_tar_seg.cpu().detach().numpy()[0, 0, :, :, :]
        # savepkl(data=(ct_aff, ct_tar_seg), path='D:/DATA/Duke/All_adult_affine/' + file_name + '.pkl')
        savepkl(data=ct_aff, path='G:/PythonProject/TransMorph/pretrain/Data/zhejiangdaxue/test/test/aff/'
                                  + file_name + '.pkl')
        idx += 1

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()