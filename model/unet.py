'''
Author: zyq
Date: 2024-05-09 10:27:28
LastEditTime: 2024-05-09 10:48:20
FilePath: /CardiacAnalysis/model/unet.py
Description: unet keras version

Copyright (c) 2024 by zyq, All Rights Reserved. 
'''

from tensorflow.keras.models import load_model
import tensorflow as tf
from base_model import BaseSegModel

# class UnetKeras:
#     def __init__(self, model_weight, frame_size=(384, 384)):
#         self.width = frame_size[0]
#         self.height = frame_size[1]
#         self.custom_objects = {'multiclass_dice': self.__multiclass_dice}
#         self.unet_model = load_model(model_weight, custom_objects=self.custom_objects)
    
#     def __multiclass_dice(self, y_true, y_pred, smooth=1e-7, num_classes=4):
#         '''
#         Multiclass Dice score. Ignores background pixel label 0
#         Pass to model as metric during compile statement
#         '''
#         y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
#         y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,1:])
#         intersect = K.sum(y_true_f * y_pred_f, axis=-1)
#         denom = K.sum(y_true_f + y_pred_f, axis=-1)
#         return K.mean((2. * intersect / (denom + smooth)))
    
#     def predict_lv_seg(self, input_img):
#         reshaped_test_img = input_img.reshape((1,self.width,self.height,1))
#         #predict
#         prediction = self.unet_model.predict(reshaped_test_img)
#         prediction = prediction.reshape([self.width, self.height, 4])
#         predict_mask = tf.math.argmax(prediction, axis = 2)
#         return predict_mask

class UnetPredictor(BaseSegModel):
    def __init__(self, frame_size=(384, 384)):
        self.width = frame_size[0]
        self.height = frame_size[1]
        self.custom_objects = {'multiclass_dice': self.__multiclass_dice}
        
    def __multiclass_dice(self, y_true, y_pred, smooth=1e-7, num_classes=4):
        '''
        Multiclass Dice score. Ignores background pixel label 0
        Pass to model as metric during compile statement
        '''
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
        y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,1:])
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))
    
    def load_model(self, model_path):
        self.model = load_model(model_path, custom_objects=self.custom_objects)
    
    def get_img_masks(self, input_img):
        reshaped_input_img = input_img.reshape((1,self.width,self.height,1))
        prediction = self.model.predict(reshaped_input_img)
        prediction = prediction.reshape([self.width, self.height, 4])
        predict_mask = tf.math.argmax(prediction, axis = 2)
        return predict_mask

if __name__ == '__main__':
    unet_predictor = UnetPredictor()
    unet_predictor.load_model('/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/model_unet_4ch.h5')
        
