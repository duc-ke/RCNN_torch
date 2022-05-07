# -*- coding: utf-8 -*-

"""
@date: 2020/4/3 下午8:07
@file: custom_bbox_regression_dataset.py
@author: zj
@description:
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import libs.util as util


class BBoxRegressionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super(BBoxRegressionDataset, self).__init__()
        self.transform = transform

        samples = util.parse_car_csv(root_dir) # data/bbox_regression/car.csv
        jpeg_list = list()
        # 각 이미지마다 하나씩 dictionary요소를 갖는 box_list 리스트 생성한다.{'image_id': ?, 'positive': ?, 'bndbox': ?}
        box_list = list()
        for i in range(len(samples)):
            sample_name = samples[i]

            jpeg_path = os.path.join(root_dir, 'JPEGImages', sample_name + '.jpg')
            bndbox_path = os.path.join(root_dir, 'bndboxs', sample_name + '.csv')
            positive_path = os.path.join(root_dir, 'positive', sample_name + '.csv')

            jpeg_list.append(cv2.imread(jpeg_path))
            bndboxes = np.loadtxt(bndbox_path, dtype=np.int, delimiter=' ')  # G.T bbox
            positives = np.loadtxt(positive_path, dtype=np.int, delimiter=' ') # positive bbox

            if len(positives.shape) == 1:
                # positive bbox와 비교하여 IoU가 가장큰 bndbox(G.T)를 반환
                bndbox = self.get_bndbox(bndboxes, positives)
                box_list.append({'image_id': i, 'positive': positives, 'bndbox': bndbox})
            else:
                for positive in positives:
                    # positive bbox와 비교하여 IoU가 가장큰 bndbox(G.T)를 반환
                    bndbox = self.get_bndbox(bndboxes, positive)
                    box_list.append({'image_id': i, 'positive': positive, 'bndbox': bndbox})

        self.jpeg_list = jpeg_list
        self.box_list = box_list

    def __getitem__(self, index: int):
        assert index < self.__len__(), '데이터셋 크기: %d, 입력된 index: %d' % (self.__len__(), index)

        # box_list(이미지별 G.T정보가 들어있는 리스트)에서 정보 추출
        box_dict = self.box_list[index]
        image_id = box_dict['image_id']
        positive = box_dict['positive']
        bndbox = box_dict['bndbox']

        jpeg_img = self.jpeg_list[image_id]  # original img(np.array)
        xmin, ymin, xmax, ymax = positive
        image = jpeg_img[ymin:ymax, xmin:xmax] # crop

        if self.transform:
            image = self.transform(image)

        # selective search의 좌표(positive region)로 pi 계산
        target = dict()
        p_w = xmax - xmin
        p_h = ymax - ymin
        p_x = xmin + p_w / 2
        p_y = ymin + p_h / 2

        # G.T bbox 좌표로 gi 계산
        xmin, ymin, xmax, ymax = bndbox
        g_w = xmax - xmin
        g_h = ymax - ymin
        g_x = xmin + g_w / 2
        g_y = ymin + g_h / 2

        # 위의 값들로 s.s 예측 bbox와 G.T bbox의 거리의 차, ti계산
        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_h
        t_w = np.log(g_w / p_w)
        t_h = np.log(g_h / p_h)

        return image, np.array((t_x, t_y, t_w, t_h))

    def __len__(self):
        return len(self.box_list)

    def get_bndbox(self, bndboxes, positive):
        """
        positive bbox와 비교하여 IoU가 가장큰 bndbox(G.T)를 반환
        :param bndboxes: [N, 4] 또는[4]
        :param positive: [4]
        :return: [4]
        """

        if len(bndboxes.shape) == 1:
            # 只有一个标注边界框，直接返回即可
            return bndboxes
        else:
            scores = util.iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]


def test():
    """
    创建数据集类实例
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = '../../data/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)

    print(data_set.__len__())
    image, target = data_set.__getitem__(10)
    print(image.shape)
    print(target)
    print(target.dtype)


def test2():
    """
    测试DataLoader使用
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = '../../data/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

    items = next(data_loader.__iter__())
    datas, targets = items
    print(datas.shape)
    print(targets.shape)
    print(targets.dtype)


if __name__ == '__main__':
    test()
    # test2()
