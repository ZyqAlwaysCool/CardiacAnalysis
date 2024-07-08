'''
Author: zyq
Date: 2024-06-19 17:12:24
LastEditTime: 2024-06-20 10:30:34
FilePath: /CardiacAnalysis/model/yolo.py
Description: yolo model

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from base_model import BaseSegModel

class YoloSegPredictor(BaseSegModel):
    def __init__(self, pred_clss=0):
        '''
        pred_clss: 需要输出的mask具体类别, 与yolo配置的分类标签有关. 此处默认0表示左心室
        '''
        self.pred_clss = pred_clss
    
    def load_model(self, model_path):
        self.model = YOLO(model_path)
    
    def get_img_masks(self, input_img):
        mask_result = None
        yolo_results = self.model.predict(input_img, retina_masks=True)
        for res in yolo_results:
            try:
                masks = res.masks.data
                boxes = res.boxes.data
                clss = boxes[:, 5]
                pred_target_idx = torch.where(clss == self.pred_clss)
                pred_target_mask = masks[pred_target_idx][0,:,:].cpu().numpy().astype(np.uint8)
                mask_result = pred_target_mask
            except:
                continue
        return mask_result

if __name__ == '__main__':
    yolo_predictor = YoloSegPredictor()
    yolo_predictor.load_model('/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/yolov8_best_v2.pt')
            
        