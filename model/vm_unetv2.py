import sys
sys.path.append('./VM_UNetV2')

import torch
from VM_UNetV2.models.vmunet.vmunet_v2 import *
from VM_UNetV2.configs.config_setting_v2 import setting_config
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from base_model import BaseSegModel


class VMUNetPredictor(BaseSegModel):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def load_model(self, model_path):
        self.model = VMUNetV2()
        best_weight = torch.load(model_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(best_weight, strict=False)
        self.model = self.model.cuda()
    
    def get_img_masks(self, input_img):
        self.model.eval()
        
        target_img = np.array(Image.open(input_img).convert('RGB'))
        msk_zeros = np.zeros(target_img.shape)
        config = setting_config
        target_img, _ = config.test_transformer((target_img, msk_zeros))
        img_reshaped = target_img.unsqueeze(0)

        pred_result = self.model(img_reshaped.cuda(non_blocking=True).float())
        pred_msk = np.where(np.squeeze(pred_result.cpu(), axis=0) > self.threshold, 1, 0)
        return pred_msk[0]
    
if __name__ == '__main__':
    img_path = '/home/kemove/zyq/giit/cardiac/unet/VM-UNetV2/data/camus_4ch/val/images/lv_img_0676.png'
    ckp_path = '/home/kemove/zyq/giit/cardiac/unet/VM-UNetV2/results/vmunet-v2_camus_Wednesday_26_June_2024_16h_53m_03s/checkpoints/best-epoch127-loss0.1203.pth'
    vmunet_predictor = VMUNetPredictor()
    vmunet_predictor.load_model(ckp_path)
    img_msk_pred = vmunet_predictor.get_img_masks(img_path)
    plt.imsave('img_msk_pred.png', img_msk_pred)


        
