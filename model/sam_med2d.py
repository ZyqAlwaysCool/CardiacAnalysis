'''
Author: zyq
Date: 2024-06-21 10:32:30
LastEditTime: 2024-06-21 11:39:38
FilePath: /CardiacAnalysis/model/sam_med2d.py
Description: sam_med2d model predictor. reference:https://github.com/OpenGVLab/SAM-Med2D/blob/main/predictor_example.ipynb

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''
import torch
import torchvision
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = '.'
CODE_DIR = f"{ROOT_DIR}/SAM_Med2D"
sys.path.append(CODE_DIR)
sys.path.append('..')

from SAM_Med2D.segment_anything import sam_model_registry
from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace
from base_model import BaseSegModel

class SAMMedPredictor(BaseSegModel):
    def __init__(self, image_size=256, encoder_adapter=True):
        self.args = Namespace()
        self.args.image_size = image_size
        self.args.encoder_adapter = encoder_adapter
    
    def load_model(self, model_path):
        self.args.sam_checkpoint = model_path
        self.model = sam_model_registry['vit_b'](self.args).to('cuda')
        self.sam_predictor = SammedPredictor(self.model)
    
    def get_img_masks(self, input_img):
        input_img = input_img * 255
        self.sam_predictor.set_image(input_img)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
            #mask_input = mask_inputs,
        )

        return masks.astype(np.uint8)[0]



# class SAMMed:
#     def __init__(self, model_weight):
#         args = Namespace()
#         args.image_size = 256
#         args.encoder_adapter = True
#         args.sam_checkpoint = model_weight
#         self.sam_model = sam_model_registry["vit_b"](args).to('cuda')
#         self.sam_predictor = SammedPredictor(self.sam_model)
    
#     def get_img_masks(self, input_img):
#         input_img = input_img * 255
#         self.sam_predictor.set_image(input_img)
#         masks, scores, logits = self.sam_predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             multimask_output=True,
#             #mask_input = mask_inputs,
#         )
        
#         return masks.astype(np.uint8)[0]

if __name__ == '__main__':
    sam_model = SAMMedPredictor()
    sam_model.load_model('/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/sam_best_v1.pth')
    print(sam_model)