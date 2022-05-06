import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


def parse_xml(xml_path):
    """
    xml(GT)경로를 받아 bbox를 파싱
    (xmin, ymin, xmax, ymax)를 np.array로 반환
    """
    # print(xml_path)
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        # print(xml_dict)

        bndboxs = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if 'car'.__eq__(obj_name) and difficult != 1:
                    bndbox = obj['bndbox']
                    bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if 'car'.__eq__(obj_name) and difficult != 1:
                bndbox = objects['bndbox']
                bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        else:
            pass

        return np.array(bndboxs)


def iou(pred_box, target_box):
    """
    전형적인 IoU 계산 방법이지만,
    target_box(GT)는 여러개(n, 4) 일수있으며 pred_box는 꼭 1개(4,) 여야만 함
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]


    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    # print('targetbox[:,0]', target_box[:, 0])
    # print('xA', xA)

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    # print('scores!', scores)  # 보통 값 한개 ex) [109]지만, GT가 여러개(n개)일 경우 n개의 np.array를 갖는다.
    return scores


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        
        # xml안 bnd가 여러개여도 최대 IoU값만 구함. selective search 갯수만큼만 IoU 갯수 리스트로 반환
        iou_list.append(max(scores))   
    return iou_list


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')
