import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('..')
sys.path.append('../model/')
sys.path.append('../model/SAM_Med2D')
sys.path.append('../utils/')
from model.model_factory import SegModelFactory
from configs.model_configs import SEG_MODEL_SETTINGS, SEG_DATASET_SETTINGS, check_dataset_valid
from utils.camus_processor import CamusProcessor
from utils.utils import LeftVentricularUtils, VisualDisplayUtils
import numpy as np
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import cv2
import shutil

class ModelEvaluator:
    def __init__(self, dataset_name: str, model_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.eval_model = SegModelFactory.get_model(model_name)
        self.eval_model.load_model(SEG_MODEL_SETTINGS[model_name]['model_info']['model_path'])
    
    def __predict_img_mask(self, input_img: np.ndarray):
        pred_mask = None
        tmp_dir = self.model_name + '_tmp'
        tmp_img = tmp_dir + '/' + 'tmp' + '_' + str(int(time.time())) + '.png'
        if self.model_name == 'unet':
            pred_mask = self.eval_model.get_img_masks(input_img)
        elif self.model_name == 'sam-med2d':
            pred_mask = self.eval_model.get_img_masks(input_img)
        elif self.model_name == 'yolov8-seg':
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            plt.imsave(tmp_img, np.squeeze(input_img), cmap='gray')
            pred_mask = self.eval_model.get_img_masks(tmp_img)
            shutil.rmtree(tmp_dir)
        elif self.model_name == 'vm-unetv2':
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            cv2.imwrite(tmp_img, input_img * 255)
            pred_mask = self.eval_model.get_img_masks(tmp_img)
            shutil.rmtree(tmp_dir)
        
        return pred_mask
    
    def __eval_camus_info(self, camus_dataset_path, model: SegModelFactory):
        '''
        camus eval function
        camus_dataset_path: camus dataset path
        model: SegModelFactory object
        '''
        pred_ef_list, true_ef_list = [], []
        
        # 1.init camus
        camus_processor = CamusProcessor(video_path=camus_dataset_path)
        all_sequences_list, all_cfg_list = camus_processor.get_all_good_sequence()
        
        cfg_idx = 0
        for video_sequence in tqdm(all_sequences_list):
            predict_mask_list = []
            for input_img in video_sequence:          
                target_mask = self.__predict_img_mask(input_img)
                predict_mask_list.append(target_mask)

            left_ventricle_utils = LeftVentricularUtils()
            pred_ef, _, _ = left_ventricle_utils.get_pred_lvef(predict_mask_list)
            true_ef = float(camus_processor.get_cfg_info(all_cfg_list[cfg_idx])['LVef'])

            pred_ef_list.append(pred_ef * 100)
            true_ef_list.append(true_ef)
            cfg_idx += 1
        
        if len(pred_ef_list) != len(true_ef_list):
            raise Exception('pred ef list len != true ef list len')
        
        error_res = []
        for idx in range(len(pred_ef_list)):
            error_percentage = abs(((pred_ef_list[idx] - true_ef_list[idx]) / true_ef_list[idx])) * 100
            error_res.append(error_percentage)
        
        def cal_percent(data, percent):
            less_than_target_percent = np.sum(data <= percent)
            total_count = len(data)
            return (less_than_target_percent / total_count) * 100
        
        error_res_data = np.array(error_res)
        less_than_5_percent = cal_percent(error_res_data, 5)
        less_than_10_percent = cal_percent(error_res_data, 10)
        less_than_15_percent = cal_percent(error_res_data, 15)
        less_than_20_percent = cal_percent(error_res_data, 20)
        
        print(f"误差低于5%的数据有: {less_than_5_percent}%")
        print(f"误差低于10%的数据有: {less_than_10_percent}%")
        print(f"误差低于15%的数据有: {less_than_15_percent}%")
        print(f"误差低于20%的数据有: {less_than_20_percent}%")

    def evaluate_model(self):
        if not check_dataset_valid(self.dataset_name):
            raise ValueError('Invalid dataset name')
        if self.dataset_name == 'camus':
            self.__eval_camus_info(SEG_DATASET_SETTINGS[self.dataset_name]['dataset_path'], self.eval_model)

if __name__ == '__main__':
    dataset = 'camus'
    model_name = 'sam-med2d'
    model_evaluator = ModelEvaluator(dataset, model_name)
    model_evaluator.evaluate_model()