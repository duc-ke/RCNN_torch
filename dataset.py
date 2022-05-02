import os

from libs.data_pascalvoc import (
    get_voc2007,
    get_custom_voc2007
)
from libs.data_finetune import get_DNN_finetune_data


    

if __name__ == '__main__':
    dir_path = 'data'

    car_root_dir = 'data/voc_car/'
    car_train_path = 'data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
    car_val_path = 'data/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'
    
    finetune_root_dir = 'data/finetune_car/'
    
    ## pascal voc 2007 원본 데이터셋 다운로드
    if not os.path.exists(dir_path):
        get_voc2007(dir_path)
    
    ## voc 2007 데이터셋에서 원하는 class 1개 골라 커스텀 데이터셋 생성
    if not os.path.exists(car_root_dir):
        get_custom_voc2007(dir_path, car_root_dir, car_train_path, car_val_path)

    ## feature extractor 부분(DNN)을 finetuning하기 위한 데이터셋 생성
    if not os.path.exists(finetune_root_dir):
        get_DNN_finetune_data(car_root_dir, finetune_root_dir)
    
    print('done')
