import os

from libs.data_pascalvoc import (
    get_voc2007,
    get_custom_voc2007
)
from libs.data_finetune import get_DNN_finetune_data
from libs.data_classifier import get_classifier_data
from libs.data_bbox_regression import get_regressor_data


if __name__ == '__main__':
    dir_path = 'data'

    car_root_dir = 'data/voc_car/'
    car_train_path = 'data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
    car_val_path = 'data/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'
    
    finetune_root_dir = 'data/finetune_car/'
    classifier_root_dir = 'data/classifier_car/'
    regressor_root_dir = 'data/bbox_regression/'
    
    ## pascal voc 2007 원본 데이터셋 다운로드(약 470Mb)
    if not os.path.exists(dir_path):
        get_voc2007(dir_path)
        print('Pascal VOC 2007 데이터셋 생성 완료.')
    
    ## voc 2007 데이터셋에서 원하는 class 1개 골라 커스텀 데이터셋 생성(약 69Mb)
    if not os.path.exists(car_root_dir):
        get_custom_voc2007(dir_path, car_root_dir, car_train_path, car_val_path)
        print('custom 데이터셋 생성 완료.')

    ## feature extractor 부분(DNN)을 finetuning하기 위한 데이터셋 생성(약 88Mb)
    if not os.path.exists(finetune_root_dir):
        get_DNN_finetune_data(car_root_dir, finetune_root_dir)
        print('Feature extracor 학습 데이터셋 생성 완료.')
    
    ## classifier 부분(SVM)을 학습하기 위한 데이터셋 생성 (약 83Mb)
    if not os.path.exists(classifier_root_dir):
        get_classifier_data(car_root_dir, classifier_root_dir)
        print('Classifier 학습 데이터셋 생성 완료.')
        
    ## bound box regressor 부분을 학습하기 위한 데이터셋 생성(약 36Mb)
    if not os.path.exists(regressor_root_dir):
        get_regressor_data(car_root_dir, finetune_root_dir, regressor_root_dir)
        print('Bnd box regressor 학습 데이터셋 생성 완료.')
    
    print('done')
