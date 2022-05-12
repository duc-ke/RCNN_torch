from distutils import extension
import os
import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import libs.selectivesearch_custom as selectivesearch

import libs.util as util


def get_device(gpu_num=0):
    """_summary_
    GPU가 있는 경우 인자로받은 번호의 GPU를 할당함
    없다면 자동으로 cpu 할당
    """
    return torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')


def get_transform():
    # inference 할 img 전처리(augmentation은 제거해야 할듯함.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(model_classifier_f, model_regressor_f, device=None):
    # classifer model load
    model_cls = alexnet()
    num_classes = 2
    num_features = model_cls.classifier[6].in_features
    model_cls.classifier[6] = nn.Linear(num_features, num_classes)
    model_cls.load_state_dict(torch.load(model_classifier_f))
    model_cls.eval()

    # weight 고정 및 device setting
    for param in model_cls.parameters():
        param.requires_grad = False
    if device:
        model_cls = model_cls.to(device)
        
    # AlexNet의 마지막 풀링레이버의 shape
    in_features = 256 * 6 * 6
    out_features = 4 # 예측하고픈 bbox coordinates
    
    # regressor model load
    model_reg = nn.Linear(in_features, out_features)
    model_reg.load_state_dict(torch.load(model_regressor_f))
    model_reg.eval()
    
    for param in model_reg.parameters():
        param.requires_grad = False
    if device:
        model_reg = model_reg.to(device)

    return model_cls, model_reg


def draw_box_with_text(img, rect_list, score_list):
    """
    openCV로 이미지에 bbox와 confidence score(from SVM) 쓰기
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def nms(rect_list, score_list):
    """
    Non-Max Suppression 구현(confidence score < 0.6, IoU > 0.3 제거대상)
    :param rect_list: list，shape:[N, 4]
    :param score_list： list，shape[N]
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # confidence score 기준 내림차순 정렬
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        # IoU
        iou_scores = util.iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # print(iou_scores)
        # 0.3이상인 selective search bbox들은 제거
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    
    # test img와 model 정의(classifier, regressor)
    test_img_path = './imgs/000334.jpg'
    model_classifier_f = './models/best_linear_svm_alexnet_car.pth'
    model_regressor_f = './models/bbox_regression_11.pth'
    gpu_idx = 6    # gpu가 없거나 1개면 0으로 설정.
    
    device = get_device(gpu_num=6)
    transform = get_transform()

    model_cls, model_reg = get_model(model_classifier_f, model_regressor_f, device=device)

    # selectivesearch
    gs = selectivesearch.get_selective_search()

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    # imgname.xml check. 없으면 bnd box 그리지 않고 pass
    anno_dirpath = os.path.dirname(test_img_path)
    anno_basename = os.path.basename(test_img_path)
    anno_base, extension = os.path.splitext(anno_basename)
    anno_file = os.path.join(anno_dirpath, (anno_base + '.xml'))
    
    if os.path.exists(anno_file):
        bndboxs = util.parse_xml(anno_file)
        for bndbox in bndboxs:
            xmin, ymin, xmax, ymax = bndbox
            cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

    selectivesearch.config(gs, img, strategy='f')
    rects = selectivesearch.get_rects(gs)
    print('selective search bbox 갯수 %d' % len(rects))

    # softmax = torch.softmax()

    svm_thresh = 0.60

    score_list = list()
    positive_list = list()

    # tmp_score_list = list()
    # tmp_positive_list = list()
    start = time.time()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        
        # SVM classifer inference
        output = model_cls(rect_transform.unsqueeze(0))[0]
        # print(f'unsqueeze shape: {rect_transform.unsqueeze(0).shape}')  # unsqueeze : 1차원 추가. [1, 3, 227, 227]
        # print(f'unsqueeze: {rect_transform.unsqueeze(0)}')
        # print(f'model out shape: {model_cls(rect_transform.unsqueeze(0)).shape}')  # [1, 2] -> [Batch(1), output(2)]
        # print(f'output shape: {output.shape}') # [2]

        if torch.argmax(output).item() == 1:
            """
            s.s bnd중 positive 예측(car) bbox들에 대하여, 0.6 confidence score만 남김
            추가로 bbox coordinate 보정값 구하기를 구현중(아직 불완전)
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # tmp_score_list.append(probs[1])
            # tmp_positive_list.append(rect)

            if probs[1] >= svm_thresh:  # 0.6 < confidence score box만 연산
                """
                # bbox regressor 개발 중
                #!!!!!!! bnd box update필요
                p_w = xmax - xmin
                p_h = ymax - ymin
                p_x = xmin + p_w / 2
                p_y = ymin + p_h / 2
                
                features = model_cls.features(rect_transform.unsqueeze(0))
                features = torch.flatten(features, 1)
                print(features.shape)  # [1, 9216]
                
                # Bbox regressor inference (구현중)
                pred_bbox = model_reg(features)[0].cpu()
                # print(pred_bbox.shape, pred_bbox)
                dp_x, dp_y, dp_w, dp_h = pred_bbox
                pred_x = p_w * dp_x + p_x
                pred_y = p_h * dp_y + p_y
                pred_w = p_w * np.exp(dp_w) 
                pred_h = p_h * np.exp(dp_h)
                
                pred_xmax = int(pred_x + pred_w/2)
                pred_xmin = int(pred_x - pred_w/2)
                pred_ymax = int(pred_y + pred_h/2)
                pred_ymin = int(pred_y - pred_h/2)
                
                # print(type(rect), rect.shape)  # np.array. (4,)
                
                # 업데이트!!
                rect = np.array([pred_xmin, pred_ymin, pred_xmax, pred_ymax])
                # print(type(rect), rect.shape)
                # exit()
                """
                score_list.append(probs[1])
                positive_list.append(rect)
                # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end - start))

    # tmp_img2 = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img2, tmp_positive_list, tmp_score_list)
    # cv2.imshow('tmp', tmp_img2)
    #
    # tmp_img = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img, positive_list, score_list)
    # cv2.imshow('tmp2', tmp_img)

    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    draw_box_with_text(dst, nms_rects, nms_scores)

    cv2.imwrite('./imgs/out.jpg', dst)
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
