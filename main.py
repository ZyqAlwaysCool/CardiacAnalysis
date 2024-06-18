'''
Author: zyq
Date: 2024-05-09 08:56:32
LastEditTime: 2024-06-18 11:42:31
FilePath: /CardiacAnalysis/main.py
Description: pipeline

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.video_processor import VideoProcessor
from utils.camus_processor import CamusProcessor
from utils.utils import LeftVentricularUtils, VisualDisplayUtils
from model.unet import UnetKeras
import argparse
from tqdm import tqdm
import sys
import numpy as np

# 评测camus数据指标误差
def eval_camus_info(camus_dataset_path, model: UnetKeras):
    pred_ef_list, true_ef_list = [], []
    
    # 1.init camus
    camus_processor = CamusProcessor(video_path=camus_dataset_path)
    all_sequences_list, all_cfg_list = camus_processor.get_all_good_sequence()
    
    cfg_idx = 0
    for video_sequence in tqdm(all_sequences_list):
        predict_mask_list = []
        for input_img in video_sequence:
            predict_mask_list.append(model.predict_lv_seg(input_img))
            if len(predict_mask_list) == 0:
                raise Exception('predict mask list is empty. check model input')
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
    



    

if __name__ == '__main__':
    # 1.parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camus', action='store_true', help='use camus dataset')
    parser.add_argument('-v', '--video', type=str, help='a video path', default='/home/kemove/zyq/giit/cardiac/CardiacAnalysis/test_video/test.avi')
    parser.add_argument('-m', '--model', type=str, help='model weight path', default='/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/model_unet_4ch.h5')
    parser.add_argument('-cp', '--camus_path', type=str, help='camus dataset path', default='/home/kemove/zyq/giit/dataset/camus-dataset/training/')
    parser.add_argument('-e', '--eval', action='store_true', help='eval model')
    args = parser.parse_args()
    
    picked_sequence_list = []
    cfg_path = None
    predict_mask_list = []
    
    # 2.load model
    if args.model is None:
        raise Exception('model weight path is required')
    unet_model = UnetKeras(args.model)
    
    #init camus
    camus_dataset_path = args.camus_path
    camus_utils = CamusProcessor(camus_dataset_path)

    # check if eval model
    if args.eval:
        eval_camus_info(camus_dataset_path, unet_model)
        sys.exit()
    
    # 3.choose dataset
    if args.camus:
        #use camus dataset
        picked_sequence_list, cfg_path = camus_utils.get_good_sequence_random()
    elif args.video is not None:
        #use video
        video_processor = VideoProcessor(args.video)
        picked_sequence_list = video_processor.preprocess_video()
    
    # 4.predict
    for input_img in picked_sequence_list:
        predict_mask_list.append(unet_model.predict_lv_seg(input_img))
    
    # 5.cal lv data
    if len(predict_mask_list) == 0:
        raise Exception('predict mask list is empty. check model input')
    left_ventricle_utils = LeftVentricularUtils()
    pred_ef, max_areas_idx, min_areas_idx = left_ventricle_utils.get_pred_lvef(predict_mask_list)

    max_left_up_x_means, max_left_down_x_means, max_left_x_means = left_ventricle_utils.get_predict_mask_x_info(predict_mask_list[max_areas_idx])
    min_left_up_x_means, min_left_down_x_means, min_left_x_means = left_ventricle_utils.get_predict_mask_x_info(predict_mask_list[min_areas_idx])
    lv_wall_amplitude = abs(max_left_x_means - min_left_x_means) #收缩末期和舒张末期相较于x轴的差值部分
    
    # 7.visualize result
    lv_info_list = left_ventricle_utils.get_lv_vol_info()
    x = [i for i in range(len(lv_info_list))]
    y = lv_info_list
    VisualDisplayUtils().gen_curve_graph(x, y, 'Left Ventricle Volume Change Info', 'video frames')
    
    # 8.conclusion
    if args.camus:
        cfg_info = camus_utils.get_cfg_info(cfg_path)
        print('camus cfg path=({})'.format(cfg_path))
        print('camus case cfg=({})'.format(cfg_info))
        true_ef = float(cfg_info['LVef'])
        pred_err = (abs(pred_ef * 100 - true_ef) / true_ef) * 100
        print('pred lvef: {:.2f}, pred_err :{:.2f}% pred lv_wall_amplitude: {:.2f} '.format(pred_ef * 100, pred_err, lv_wall_amplitude))
    else:
        print('pred lvef: {:.2f}, pred lv_wall_amplitude: {:.2f} '.format(pred_ef * 100, lv_wall_amplitude))