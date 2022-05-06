import time
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet

from libs.custom_classifier_dataset import CustomClassifierDataset
from libs.custom_hard_negative_mining_dataset import CustomHardNegativeMiningDataset
from libs.custom_batch_sampler import CustomBatchSampler
from libs.util import check_dir
from libs.util import save_model

batch_positive = 32
batch_negative = 96
batch_total = 128


def load_data(data_root_dir):
    """_summary_
    데이터 dir로부터 train, validation dataset_loader를 리턴.
    - transform 적용(resize, flip, normalization)
    selective search region을 positve, negative로 나눈뒤, 
    1차, positive: negative s.s region을 1:1로 맞추고,
    2차, 128배치(양:음 = 32:96 갯수)단위로 데이터 준비 (negative 비율이 너무 많아서 강제 설정)
    
    이렇게 1차를 1:1로 맞추지 않을 경우 sampler가 복원 추출이므로 negative를 처음 모델에 보여줄때,
    positive는 여러번 모델에 전해주게 되어 편향 모델이 만들어 질것이므로. 추가 스텝을 적용한 한듯하다.
    
    또한 1:1로 맞추면서 남은 negative은 따로 모았다가 "hard negative mining" 기법으로 학습하는데 쓰인다.
    hard negative : 네거티브 샘플이라고 말하기 어려운 샘플. 즉, 실제 neg인데 pos라고 예측하기 쉬운 데이터이다. FP샘플들이 해당한다.
    hard negative mining : FP샘플들을 모으는 것이다. 이것들을 training 하면 FP 오류가 줄어든다. precision이 높아지겠지.
    """
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)  # data/classifier_car/train

        data_set = CustomClassifierDataset(data_dir, transform=transform)
        # print(len(data_set))  # 358,919개의 selective search regions
        # print(data_set[0])  # 튜플로 묶여있음 : (tensor(3, 227, 227), int(0또는 1), {'rect':np.arr(1,2,3,4), 'image_id':1})
        # print(type(data_set[0][0]), type(data_set[0][1]), data_set[0][0].shape)  # <class 'torch.Tensor'> <class 'int'> torch.Size([3, 227, 227])
        
        if name == 'train':
            """
            초기 pos:neg 비율은 1:1. 양성 샘플 수가 훨씬 적기 때문에 맞춰주고 32:96으로 2차 sampling
            """
            
            # positive s.s region의 리스트(딕셔러니 요소), [{'rect':np.arr(1,2,3,4), 'image_id':1}, {...}]
            positive_list = data_set.get_positives()  
            negative_list = data_set.get_negatives()
            # print(len(positive_list), len(negative_list))  # selective search region (625, 358294) 

            ## positive s.s region 갯수만큼 negative s.s region 뽑아줌
            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
            # 선별한 negative s.s region 정보
            init_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if idx in init_negative_idxs]
            # 남은 negative s.s region 정보
            remain_negative_list = [negative_list[idx] for idx in range(len(negative_list))
                                    if idx not in init_negative_idxs]

            # positive s.s region와 갯수가 맞도록 선별한 negative sample(s.s region)를 새로 업데이트 
            data_set.set_negative_list(init_negative_list)
            data_loaders['remain'] = remain_negative_list
            
            # print(data_set.get_positive_num(), data_set.get_negative_num()) # 양성 샘플과 음성 샘플 갯수 맞줘짐. 625, 625

        # 128배치 단위로 뽑아줄때 positive s.s region 32, negative s.s region 96개로 맞춰서 뽑도록 샘플러 설정
        sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)
        # print(len(sampler)) # 1152 (9*128)

        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader
        data_sizes[name] = len(sampler)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))


def get_hard_negatives(preds, cache_dicts):
    """
    1:1로 맞추면서 남은 negative은 따로 모았다가 "hard negative mining" 기법으로 학습하는데 쓰인다.
    hard negative : 네거티브 샘플이라고 말하기 어려운 샘플. 즉, 실제 neg인데 pos라고 예측하기 쉬운 데이터이다. FP샘플들이 해당한다.
    hard negative mining : FP샘플들을 모으는 것이다. 이것들을 training 하면 FP 오류가 줄어든다. precision이 높아지겠지.
    """
    fp_mask = preds == 1 # neg샘플이니 pos판별시 FP(False Positive)
    tn_mask = preds == 0 # TN(True Negative)

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negatie_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negatie_list


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num: {} - data size: {}'.format(
                phase, data_set.get_positive_num(), data_set.get_negative_num(), data_sizes[phase]))

            # Iterate 시작
            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(f'input shape {inputs.shape}') # (128, 3, 227, 227)
                    # print(f'output shape {outputs.shape}') # (128, 2)
                    
                    _, preds = torch.max(outputs, 1)
                    # print(f'pred shape {preds.shape}')  #(128)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)  # 특이하게 loss * 배치크기를 한다.(hinge를 알아야 이해할듯)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # 아래는 hard negative mining 기법을 적용하기 위한 코드들임.
        # 즉, neg 샘플들 중 FP를 찾아내어 데이터셋을 업데이트 하는 것이다.
        train_dataset = data_loaders['train'].dataset
        remain_negative_list = data_loaders['remain']  # [{'rect':np.arr(1,2,3,4), 'image_id':1}, {...}]
        jpeg_images = train_dataset.get_jpeg_images()
        transform = train_dataset.get_transform()

        with torch.set_grad_enabled(False):  # 여기부턴 weight update 안함
            
            # 위의 load_data 함수 부분에서 1:1 pos:neg로 맞추고 남겨놨던 샘플들을 data_loader로 만든다.
            remain_dataset = CustomHardNegativeMiningDataset(remain_negative_list, jpeg_images, transform=transform)
            remain_data_loader = DataLoader(remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True)

            # 기존 neg regions들을 가져온다.
            negative_list = train_dataset.get_negatives()
            # data_loaders(dictionary)에 add_negative key에 해당하는 value(리스트)가져온다. 없으면 빈 리스트 반환
            add_negative_list = data_loaders.get('add_negative', [])

            running_corrects = 0
            # Iterate over data.
            # cache_dicts는 remain_data_loader의 정보 이다 : {'rect':np.arr(1,2,3,4), 'image_id':1}
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                # print(outputs.shape)
                _, preds = torch.max(outputs, 1)  # shape: (128)

                running_corrects += torch.sum(preds == labels.data)

                # FP negative region들만 추린다.
                hard_negative_list, easy_neagtive_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                # negative_list와 add_negative_list에 FP를 넣는다.
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)

            remain_acc = running_corrects.double() / len(remain_negative_list)
            print('remiam negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))

            ## hard negatives mining 기법을 적용한다.
            # neg sample을 train dataset에 업데이트 하고 data_loader를 다시 불러 온다. iter가 늘게 된다.
            train_dataset.set_negative_list(negative_list) 
            tmp_sampler = CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                             batch_positive, batch_negative)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            data_loaders['add_negative'] = add_negative_list
            data_sizes['train'] = len(tmp_sampler)

        # epoch 마다 모델을 저장한다.
        save_model(model, 'models/linear_svm_alexnet_car_%d.pth' % epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loaders, data_sizes = load_data('./data/classifier_car')

    model_path = './models/alexnet_car.pth'  # weight만 저장되어 있음.
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))  # Feature extractor finetuning weight로 교체
    model.eval()
    
    

    
    # SVM이 아닌 Feature extractor 부분은 Freeze(weight update 못하게 막음)
    for param in model.parameters(): # 모델내 존재하는 모든 weight들을 부름
        param.requires_grad = False
    
    # # Tip. 이렇게 하면 이름과 함께 param 나옴
    # for name, param in model.named_parameters():
    #     print(name)
    
    # classifier[6]에 Linear fc를 주면서 Linear SVM을 적용 by hinge loss
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # print(model)
    model = model.to(device)

    criterion = hinge_loss
    # Momentum opt + scheduler
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # epoch 4마다 lr를 줄여줌 (pre lr * gamma)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=10, device=device)
    exit()
    # classifier (SVM) 모델 저장
    save_model(best_model, 'models/best_linear_svm_alexnet_car.pth')
