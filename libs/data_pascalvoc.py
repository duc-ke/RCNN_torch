import os
import shutil
import random
import numpy as np
import cv2
import xmltodict
from torchvision.datasets import VOCDetection

from libs.util import check_dir
# from util import check_dir   # __main__ 실행시 경로 수정

suffix_xml = '.xml'
suffix_jpeg = '.jpg'

def get_voc2007(dir_path, is_return=False):
    """_summary_
    voc 2007 데이터셋을 torchvision으로부터 다운받는다.
    """
    dataset = VOCDetection(dir_path, year='2007', image_set='trainval', download=True)
    
    if is_return:
        return dataset

def parse_train_val(data_path):
    """
    'data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt' 데이터셋에서
    positive sample만 추려 아이디 저장
    """
    samples = []

    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            # print(res)  # ['009949', '-1'] ['009959', '', '1']
            if len(res) == 3 and int(res[2]) == 1:  # 1: positive sample(차 있음), -1: negative sample(차 없음)
                samples.append(res[0])

    return np.array(samples)


def sample_train_val(samples):
    """
    쓰지않는 함수.
    """
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)

        random_samples = random.sample(range(length), int(length / 10))
        # print(random_samples)
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


def save_car(
    car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir,
    voc_ori_dir
    ):
    """
    지정한 경로에 train, val 각각 annotation(xml)과 image(jpg) voc2007 원본으로 부터 복사.
    """
    
    voc_annotation_dir = os.path.join(voc_ori_dir, 'VOCdevkit/VOC2007/Annotations/')
    voc_jpeg_dir = os.path.join(voc_ori_dir, 'VOCdevkit/VOC2007/JPEGImages/')
    
    for sample_name in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')

def get_custom_voc2007(voc_ori_dir, car_root_dir, car_train_path, car_val_path):
    """_summary_
    제공된 train_path, val_path에서 positive sample(img)만 추려 새로운 dir로 이동
    """
    samples = {'train': parse_train_val(car_train_path), 'val': parse_train_val(car_val_path)}
    print(samples)
    
    check_dir(car_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)
        save_car(
            samples[name], data_root_dir, 
            data_annotation_dir, data_jpeg_dir,
            voc_ori_dir
        )
        

if __name__ == '__main__':
    car_train_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
    parse_train_val(car_train_path)