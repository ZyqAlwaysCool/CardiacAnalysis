'''
Author: zyq
Date: 2024-05-09 08:56:32
LastEditTime: 2024-05-09 17:26:33
FilePath: /CardiacAnalysis/main.py
Description: pipeline

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.video_processor import VideoProcessor
from utils.camus_processor import CamusProcessor
from utils.utils import LeftVentricularUtils
from model.unet import UnetKeras
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    # 1.parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camus', action='store_true', help='use camus dataset')
    parser.add_argument('-v', '--video', type=str, help='a video path', default='/home/kemove/zyq/giit/dataset/test.avi')
    parser.add_argument('-m', '--model', type=str, help='model weight path', default='/home/kemove/zyq/giit/cardiac/CAMUS-dataset/model/model_unet_4ch.h5')
    parser.add_argument('-cp', '--camus_path', type=str, help='camus dataset path', default='/home/kemove/zyq/giit/dataset/camus-dataset/training/')
    args = parser.parse_args()
    
    picked_sequence_list = []
    cfg_path = None
    predict_mask_list = []
    
    # 2.load model
    if args.model is None:
        raise Exception('model weight path is required')
    unet_model = UnetKeras(args.model)

    # 3.choose dataset
    if args.camus:
        #use camus dataset
        camus_dataset_path = args.camus_path
        camus_utils = CamusProcessor(camus_dataset_path)
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
    lv_wall_amplitude = abs(max_left_x_means - min_left_x_means)
    
    
    # 6.conclusion
    if args.camus:
        cfg_info = camus_utils.get_cfg_info(cfg_path)
        print('camus cfg path=({})'.format(cfg_path))
        print('camus case cfg=({})'.format(cfg_info))

    print('pred lvef: {:.2f}, pred lv_wall_amplitude: {:.2f}'.format(pred_ef * 100, lv_wall_amplitude))