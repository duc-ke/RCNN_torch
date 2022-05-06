import os
import shutil
import numpy as np
import libs.util as util

# ## __main__ 실행시, 경로 수정한 라이브러리 활성화 필요.
# import util

def get_regressor_data(custom_dir, feature_extracor_dir, to_dir):
    """_summary_
    voc_car/train 에서 G.T bbox coordinate 추출
    finetune_car 에서 positive bbox(IOU>0.5)추출하여 그중 0.6인 bbox를 추가로 추출
    bbox-regressor는 분류문제가 아니므로 negative sample 또는 bbox들은 제거하고 
    positive bbox(0.6<IOU)만 추려서 G.T bbox와 함께 디렉토리 구성
    """
    voc_car_train_dir = os.path.join(custom_dir, 'train')
     # ground truth
    gt_annotation_dir = os.path.join(voc_car_train_dir, 'Annotations')
    jpeg_dir = os.path.join(voc_car_train_dir, 'JPEGImages')

    classifier_car_train_dir = os.path.join(feature_extracor_dir, 'train')
    # positive
    positive_annotation_dir = os.path.join(classifier_car_train_dir, 'Annotations')
    dst_root_dir = to_dir
    dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
    dst_bndbox_dir = os.path.join(dst_root_dir, 'bndboxs')
    dst_positive_dir = os.path.join(dst_root_dir, 'positive')
    
    util.check_dir(dst_root_dir)
    util.check_dir(dst_jpeg_dir)
    util.check_dir(dst_bndbox_dir)
    util.check_dir(dst_positive_dir)
    
    print(voc_car_train_dir)  # voc_car/train
    print(gt_annotation_dir)  # voc_car/train/Annotations
    print(classifier_car_train_dir) # finetune_car/train
    print(positive_annotation_dir) # finetune_car/train/Annotations

    # car가 들어간 이미지 리스트(voc_car/train/car.csv)
    samples = util.parse_car_csv(voc_car_train_dir)
    res_samples = list()  # finetune_car에서도 positive sample(img)로 거른 샘플리스트
    total_positive_num = 0
    for sample_name in samples:
        # finetune_car/train/Annotations 안의 _1.csv는 positive bbox로 추출했었음 : IoU>=0.5 기준
        positive_annotation_path = os.path.join(positive_annotation_dir, sample_name + '_1.csv')
        positive_bndboxes = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
        # print(positive_bndboxes.shape) # bbox 2차원 리스트, (339, 4) - selective search
        # 해당 샘플의 GT는 /voc_car/train/Annotations/*.xml에 있음.
        gt_annotation_path = os.path.join(gt_annotation_dir, sample_name + '.xml')
        bndboxs = util.parse_xml(gt_annotation_path)
        print(bndboxs.shape) # G.T의 bbox 2차원 리스트, (1, 4)
        
        # s.s의 bbox가 여러개일때 G.T의 bbox들과 비교해서 0.6 IoU이상이면 해당 bbox만 살리고
        # 나머지 bbox는 모두 지움.
        positive_list = list()
        if len(positive_bndboxes.shape) == 1 and len(positive_bndboxes) != 0:
            scores = util.iou(positive_bndboxes, bndboxs)
            if np.max(scores) > 0.6:
                positive_list.append(positive_bndboxes)
        elif len(positive_bndboxes.shape) == 2:
            for positive_bndboxe in positive_bndboxes:
                scores = util.iou(positive_bndboxe, bndboxs)
                if np.max(scores) > 0.6:
                    positive_list.append(positive_bndboxe)
        else:
            pass

        # positive bbox들이 1개라도 있다면, jpg, annotation, 
        if len(positive_list) > 0:
            # positive 이미지 : bbox_regression/JPEGImages
            jpeg_path = os.path.join(jpeg_dir, sample_name + ".jpg")
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")
            shutil.copyfile(jpeg_path, dst_jpeg_path)
            # G.T bbox : bbox_regression/bndboxs
            dst_bndbox_path = os.path.join(dst_bndbox_dir, sample_name + ".csv")
            np.savetxt(dst_bndbox_path, bndboxs, fmt='%s', delimiter=' ')
            # positive s.s bbox : bbox_regression/positive
            dst_positive_path = os.path.join(dst_positive_dir, sample_name + ".csv")
            np.savetxt(dst_positive_path, np.array(positive_list), fmt='%s', delimiter=' ')

            total_positive_num += len(positive_list)
            res_samples.append(sample_name)
            print('save {} done'.format(sample_name))
        else:
            print('-------- {} '.format(sample_name))

    # positive bbox가 있는 샘플만 car.csv로 저장
    dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
    np.savetxt(dst_csv_path, res_samples, fmt='%s', delimiter=' ')
    print('total positive num: {}'.format(total_positive_num))

if __name__ == '__main__':
    car_root_dir = '../data/voc_car/'
    finetune_root_dir = '../data/finetune_car/'
    regressor_root_dir = '../data/bbox_regression/'
    get_regressor_data(car_root_dir, finetune_root_dir, regressor_root_dir)
