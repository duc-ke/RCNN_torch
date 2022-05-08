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


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def get_model(device=None):
    # classifer model load
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    # weight 고정 및 device setting
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)
        
    # AlexNet의 마지막 풀링레이버의 shape
    in_features = 256 * 6 * 6
    out_features = 4 # 예측하고픈 bbox coordinates
    model_linear = nn.Linear(in_features, out_features)
    
    for param in model_linear.parameters():
        param.requires_grad = False
    if device:
        model_linear = model_linear.to(device)

    return model, model_linear


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
    device = get_device()
    transform = get_transform()
    model, model_linear = get_model(device=device)

    # selectivesearch
    gs = selectivesearch.get_selective_search()

    # test_img_path = '../imgs/000007.jpg'
    # test_xml_path = '../imgs/000007.xml'
    test_img_path = './imgs/000318.jpg'
    test_xml_path = './imgs/000318.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    bndboxs = util.parse_xml(test_xml_path)
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
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            """
            s.s bnd중 positive 예측(car) bbox들에 대하여, 0.6 confidence score만 남김
            추가로 bbox coordinate 보정값 구하기를 구현중(아직 불완전)
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # tmp_score_list.append(probs[1])
            # tmp_positive_list.append(rect)

            if probs[1] >= svm_thresh:
                #!!!!!!! bnd box update필요
                p_w = xmax - xmin
                p_h = ymax - ymin
                p_x = xmin + p_w / 2
                p_y = ymin + p_h / 2
                
                features = model.features(rect_transform.unsqueeze(0))
                features = torch.flatten(features, 1)
                print(features.shape)  # [1, 9216]
                
                # Bbox regressor inference (구현중)
                pred_bbox = model_linear(features)[0].cpu()
                # print(pred_bbox.shape, pred_bbox)
                dp_x, dp_y, dp_w, dp_h = pred_bbox
                pred_x = p_w * dp_x + p_x
                pred_y = p_h * dp_y + p_y
                pred_w = p_w * np.exp(dp_x) 
                pred_h = p_h * np.exp(dp_x)
                
                pred_xmax = int(pred_x + pred_w/2)
                pred_xmin = int(pred_x - pred_w/2)
                pred_ymax = int(pred_y + pred_h/2)
                pred_ymin = int(pred_y - pred_h/2)
                
                print(type(rect), rect.shape)
                
                # 업데이트!!
                rect = np.array([pred_xmin, pred_ymin, pred_xmax, pred_ymax])
                # print(type(rect), rect.shape)
                # exit()
                
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
