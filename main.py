'''
Author: zyq
Date: 2024-05-09 08:56:32
LastEditTime: 2024-06-21 16:57:25
FilePath: /CardiacAnalysis/main.py
Description: pipeline

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
from utils.video_processor import VideoProcessor
from utils.camus_processor import CamusProcessor
from utils.utils import LeftVentricularUtils, VisualDisplayUtils
sys.path.append('./model/')
sys.path.append('./model/SAM_Med2D')
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
from model.model_factory import SegModelFactory
from configs.model_configs import SEG_MODEL_SETTINGS
import time


# 输入图像序列, 输出mask矩阵序列, 基于不同模型做预测
def predict_img_mask(input_img, model_name, pred_model):
    pred_mask = None
    tmp_dir = model_name + '_tmp'
    tmp_img = tmp_dir + '/' + 'tmp' + '_' + str(int(time.time())) + '.png'
    if model_name == 'unet':
        pred_mask = pred_model.get_img_masks(input_img)
    elif model_name == 'sam-med2d':
        pred_mask = pred_model.get_img_masks(input_img)
    elif model_name == 'yolov8-seg':
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        plt.imsave(tmp_img, np.squeeze(input_img), cmap='gray')
        pred_mask = pred_model.get_img_masks(tmp_img)
        shutil.rmtree(tmp_dir)
    elif model_name == 'vm-unetv2':
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        cv2.imwrite(tmp_img, input_img * 255)
        pred_mask = pred_model.get_img_masks(tmp_img)
        shutil.rmtree(tmp_dir)
    
    return pred_mask

if __name__ == '__main__':
    # 1.parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camus', action='store_true', help='use camus dataset')
    parser.add_argument('-v', '--video', type=str, help='a video path', default='/home/kemove/zyq/giit/cardiac/CardiacAnalysis/test_video/test.avi')
    parser.add_argument('-w', '--weight', type=str, help='model weight', default=None)
    parser.add_argument('-m', '--model', type=str, help='model name', default='unet')
    parser.add_argument('-cp', '--camus_path', type=str, help='camus dataset path', default='/home/kemove/zyq/giit/dataset/camus-dataset/training/')
    parser.add_argument('-e', '--eval', action='store_true', help='eval model')
    args = parser.parse_args()
    
    picked_sequence_list = []
    camus_cfg_path = None
    predict_mask_list = []
    camus_utils = None
    
    model_name = args.model
    pred_model = SegModelFactory.get_model(model_name)
    if args.weight is None:
        pred_model.load_model(SEG_MODEL_SETTINGS[model_name]['model_info']['model_path'])
    else:
        pred_model.load_model(args.weight)
    
    # 1.choose dataset
    if args.camus:
        #use camus dataset
        camus_utils = CamusProcessor(args.camus_path)
        if args.model == 'unet':
            picked_sequence_list, camus_cfg_path = camus_utils.get_good_sequence('0044') #选择指定的good病例
        else:
            picked_sequence_list, camus_cfg_path = camus_utils.get_good_sequence_random() #随机选择一个good的病例序列帧
    elif args.video is not None:
        #use user video
        video_processor = VideoProcessor(args.video)
        picked_sequence_list = video_processor.preprocess_video()
    
    if len(picked_sequence_list) == 0:
        raise Exception('no video sequence found')
    
    # 2.predict. 模型推理可能出现err, 比如图像中没有目标物体, 此时选择跳过当前图像, 继续做下一张图像推理
    for input_img in picked_sequence_list:
        try:
            predict_mask_list.append(predict_img_mask(input_img, model_name, pred_model))
        except:
            continue
    
    # 3.cal lv info
    if len(predict_mask_list) == 0:
        raise Exception('predict mask list is empty. check model input')
    left_ventricle_utils = LeftVentricularUtils()
    pred_ef, max_areas_idx, min_areas_idx = left_ventricle_utils.get_pred_lvef(predict_mask_list)
    
    max_left_up_x_means, max_left_down_x_means, max_left_x_means = left_ventricle_utils.get_predict_mask_x_info(predict_mask_list[max_areas_idx])
    min_left_up_x_means, min_left_down_x_means, min_left_x_means = left_ventricle_utils.get_predict_mask_x_info(predict_mask_list[min_areas_idx])
    lv_wall_amplitude = abs(max_left_x_means - min_left_x_means) #收缩末期和舒张末期相较于x轴的差值部分

    # 4.visualize result
    lv_info_list = left_ventricle_utils.get_lv_vol_info()
    x = [i for i in range(len(lv_info_list))]
    y = lv_info_list
    VisualDisplayUtils().gen_curve_graph(x, y, 'Left Ventricle Volume Change Info', 'video frames')

    # 5.conclusion
    if args.camus:
        cfg_info = camus_utils.get_cfg_info(camus_cfg_path)
        print('camus cfg path=({})'.format(camus_cfg_path))
        print('camus case cfg=({})'.format(cfg_info))
        true_ef = float(cfg_info['LVef'])
        pred_err = (abs(pred_ef * 100 - true_ef) / true_ef) * 100
        print('pred lvef: {:.2f}, pred_err :{:.2f}% pred lv_wall_amplitude: {:.2f} '.format(pred_ef * 100, pred_err, lv_wall_amplitude))
    else:
        print('pred lvef: {:.2f}, pred lv_wall_amplitude: {:.2f} '.format(pred_ef * 100, lv_wall_amplitude))