# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    openCV의 selective search 실행. bbox coordinates를 받는다.
    [[xmin, ymin, xmax, ymax][]..] 로 구성.
    """
    gs = get_selective_search()

    img = cv2.imread('../data/voc_car/train/JPEGImages/000012.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects.shape)  # (4647, 4) 4개의 coordinate, 4647 objs
    print(rects)  # [[329  20 344  33] [185 268 459 333] ..
