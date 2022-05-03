import os
from re import S
import time
import shutil
import numpy as np
import cv2
import libs.selectivesearch_custom as selectivesearch

from libs.util import check_dir
from libs.util import parse_car_csv
from libs.util import parse_xml
from libs.util import compute_ious

# ## __main__ 실행시, 경로 수정한 라이브러리 활성화 필요.
# import selectivesearch_custom as selectivesearch
# from util import check_dir
# from util import parse_car_csv
# from util import parse_xml
# from util import compute_ious



# train
# positive num: 66517
# negatie num: 464340
# val
# positive num: 64712
# negative num: 415134

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
    # print(type(rects), type(bndboxs))  # 둘다 numpy.array
    # print(rects.shape, bndboxs.shape)  # (4647, 4), (1, 4)

    maximum_bndbox_size = 0  # GT(xml)에서 가장큰 bbox 넓이(가로 x 세로) 계산
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    iou_list = compute_ious(rects, bndboxs)
    # print(np.array(iou_list).shape)  # (4647, ) selective search 만큼의 갯수만 나옴

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)  # s.s의 bnd box 넓이 계산

        iou_score = iou_list[i]
        """
        Feature extractor fine tuning을 위한 샘플링
          - posivie sample : IoU 0.5 이상인 s.s bbox
          - negative sample : IoU 0.5 미만 + G.T bbox 넓이의 1/5 이상인 s.s bbox
        """
        if iou_score >= 0.5:
            positive_list.append(rects[i])
        if 0 < iou_score < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass

    # bbox 좌표 반환
    return positive_list, negative_list


def get_DNN_finetune_data(from_dir, to_dir):
    """_summary_
    [Feature Extractor(AlexNet) finetuning에 이용할 데이터셋을 만든다.]
    기존 custom VOC폴더에서 postive, negative sample을 추출, 새로운 폴더에 옮긴다.
    """
    car_root_dir = from_dir
    finetune_root_dir = to_dir
    
    check_dir(finetune_root_dir)

    gs = selectivesearch.get_selective_search()  # openCV의 selectivesearch 객체 생성
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)   # data/voc_car/train
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(finetune_root_dir, name)   # data/finetune_car/train
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)   # data/voc_car/train/car.csv 에 있던 positive samples 리스트
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        
        for sample_name in samples:  # positive 샘플(즉, train/val 모든 이미지 처리)
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            
            # selective search bbox와 G.T bbox를 비교하여 positive/negativa bbox 리스트 반환
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')  # ex) 000012_1.csv
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
    finetune_root_dir = '../data/finetune_car/'
    get_DNN_finetune_data(car_root_dir, finetune_root_dir)
