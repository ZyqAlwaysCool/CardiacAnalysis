SEG_MODEL_SETTINGS = {
    'yolov8-seg': {
        'model_info': {
            'model_name': 'yolov8-seg',
            'model_path': '/home/kemove/zyq/giit/cardiac/CardiacAnalysis/checkpoints/yolov8_best_v2.pt', # pretrained model
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train24/weights/best.pt', # unetencoder maxpool
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train23/weights/best.pt', # unetencoder nomaxpool
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train25/weights/best.pt', # origin model no pretrained weight
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train26/weights/best.pt', # yolov8x-unet-encoder-maxpool2d
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train37/weights/best.pt', # origin model scale-x
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train38/weights/best.pt', # large camus, c3str
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train40/weights/best.pt' # large-camus, yolov8x-focus-c3str-unet-encoder
            
            #train all in pretrained
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train18/weights/best.pt', # yolov8s_GAM_seg
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train19/weights/best.pt', # yolov8s-seg
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train49/weights/best.pt', # yolov8x-unetencoder-pretrained amp=False
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train4/weights/best.pt', # yolov8x-swin-seg-pretrained amp=True
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train3/weights/best.pt', # yolov8x-swin-seg-pretrained amp=False
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train50/weights/best.pt', # yolov8x-focus-c3str-pretrained amp=False
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train61/weights/best.pt', # yolov8x-focus-c3str-unet-encoder amp=False
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train60/weights/best.pt' # yolov8x-unet-encoder-focus-swin-sppc-seg amp=False
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train5/weights/best.pt' # yolov8x-c2-seg amp=False
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train6/weights/best.pt' # yolov8x-c2-dwconv-seg amp=False
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train65/weights/best.pt' # yolov8x-focus-swin-sppc-seg
            #'model_path': '/home/kemove/zyq/giit/cardiac/yolo/runs/segment/train68/weights/best.pt', # yolov8x-p6-trans-c2-seg
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train4/weights/best.pt', # Yolov8xswin_seg

            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train8/weights/best.pt', # yolov8x_DW_C3TR_focus_seg
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train9/weights/best.pt', # yolov8x_DW_swin_FOCUS-3_seg
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train12/weights/best.pt', # yolov8x_DW_swin_FOCUS2_seg
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train11/weights/best.pt', # yolov8x_DW_FOCUS_seg.yaml
            #'model_path': '/home/kemove/hyx/ultralytics_STF/runs/segment/train22/weights/best.pt', # yolov8x_DW_FOCUS_sppc_seg.yaml
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
