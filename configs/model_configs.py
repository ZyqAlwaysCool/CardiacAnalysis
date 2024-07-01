SEG_MODEL_SETTINGS = {
    'yolov8-seg': {
        'model_info': {
            'model_name': 'yolov8-seg',
            'model_path': '/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/yolov8_best_v2.pt',
        }
    },
    'sam-med2d': {
        'model_info': {
            'model_name': 'sam-med2d',
            'model_path': '/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/sam_best_v1.pth',
        }
    },
    'unet': {
        'model_info': {
            'model_name': 'unet',
            'model_path': '/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/model_unet_4ch.h5',
        }
    },
    'vm-unetv2': {
        'model_info': {
            'model_name': 'vm-unetv2',
            'model_path': '/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/vm_unetv2_best_v1.pth',
        }
    },
}

SEG_DATASET_SETTINGS = {
    'camus': {
        'dataset_path': '/home/kemove/zyq/giit/dataset/camus-dataset/training/'
    }
}

def check_model_valid(model_name):
    if model_name not in SEG_MODEL_SETTINGS.keys():
        return False
    return True

def check_dataset_valid(dataset_name):
    if dataset_name not in SEG_DATASET_SETTINGS.keys():
        return False
    return True
