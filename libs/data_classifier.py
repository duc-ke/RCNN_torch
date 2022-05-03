import random
import numpy as np
import shutil
import time
import cv2
import os
import xmltodict
import libs.selectivesearch_custom as selectivesearch

from libs.util import check_dir
from libs.util import parse_car_csv
from libs.util import parse_xml
from libs.util import iou
from libs.util import compute_ious


# # ## __main__ 실행시, 경로 수정한 라이브러리 활성화 필요.
# from util import check_dir
# from util import parse_car_csv
# from util import parse_xml
# from util import iou
# from util import compute_ious

# train
# positive num: 625
# negative num: 366028
# val
# positive num: 625
# negative num: 321474

def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    한 샘플의 jpg(img)와 xml(anno)를 받음
      - jpg : selective search 후 coordinates를 2차원 numpy(ex) 4000, 4 shape)으로 받음 
      - xml : GT bbox coordinates(좌상단, 우하단)을 파싱.
    selective search수행 후 나온 region proposals들을 가지고 G.T bbox와 비교하여 positive, negative bbox 좌표 반환
    """
    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q')
    rects = selectivesearch.get_rects(gs)
    bndboxs = parse_xml(annotation_path)

    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # 
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        """
        Classifier 학습을 위한 데이터 샘플링
          - posivie sample : 모든 G.T bbox
          - negative sample : IoU 0.3 미만 + G.T bbox 넓이의 1/5 이상인 s.s bbox
        """
        if 0 < iou_score <= 0.3 and rect_size > maximum_bndbox_size / 5.0:
            # 负样本
            negative_list.append(rects[i])
        else:
            pass

    return bndboxs, negative_list


def get_classifier_data(from_dir, to_dir):
    """_summary_
    [Classifer (SVM) 학습에 이용할 데이터셋을 만든다.]
    기존 custom VOC폴더에서 postive, negative sample을 추출, 새로운 폴더에 옮긴다.
    """
    car_root_dir = from_dir
    classifier_root_dir = to_dir
    
    check_dir(classifier_root_dir)

    gs = selectivesearch.get_selective_search()
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)    # data/voc_car/train
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(classifier_root_dir, name)  # data/classifier_car/train
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)
        # 
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            # selective search bbox와 G.T bbox를 비교하여 positive/negativa bbox 리스트 반환
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # bbox좌표(xmin, ymin, xmax, ymax)를 annotation(.csv)으로 저장
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))

if __name__ == '__main__':
    car_root_dir = '../data/voc_car/'
    classifier_root_dir = '../data/classifier_car/'
    get_classifier_data(car_root_dir, classifier_root_dir)