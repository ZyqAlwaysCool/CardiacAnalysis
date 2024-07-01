from sam_med2d import SAMMedPredictor
from unet import UnetPredictor
from vm_unetv2 import VMUNetPredictor
from yolo import YoloSegPredictor
from configs.model_configs import check_model_valid, SEG_MODEL_SETTINGS

class SegModelFactory:
    @staticmethod
    def get_model(model_name):
        if not check_model_valid(model_name):
            raise ValueError(f'{model_name} is not a valid model name. support model list=({SEG_MODEL_SETTINGS.keys()})')
        if model_name == 'sam-med2d':
            return SAMMedPredictor()
        elif model_name == 'unet':
            return UnetPredictor()
        elif model_name == 'yolov8-seg':
            return YoloSegPredictor()
        elif model_name == 'vm-unetv2':
            return VMUNetPredictor()
    
    
if __name__ == '__main__':
    model = SegModelFactory.get_model('yolov8-seg')
    model.load_model('/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/yolov8_best_v2.pt')