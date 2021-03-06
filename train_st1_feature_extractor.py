import os
import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from libs.custom_finetune_dataset import CustomFinetuneDataset
from libs.custom_batch_sampler import CustomBatchSampler
from libs.util import check_dir


def load_data(data_root_dir):
    """_summary_
    데이터 dir로부터 train, validation dataset_loader를 리턴.
    - transform 적용(resize, flip, normalization)
    selective search region을 positve, negative로 나눈뒤, 
    128배치(양:음 = 32:96 갯수)단위로 데이터 준비 (negative 비율이 너무 많아서 강제 설정)
    """
    # 이미지크기조정 + augmentation(flip) + normalization 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)  # data/finetune_car/train
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        # print(len(data_set))  # 520,954개의 selective search regions
        # print(data_set[0])  # 튜플로 묶여있음 : (tensor(3, 227, 227), int(0또는 1))
        # print(type(data_set[0][0]), type(data_set[0][1]), data_set[0][0].shape)  # <class 'torch.Tensor'> <class 'int'> torch.Size([3, 227, 227])
        
        # print(data_set.get_positive_num(), data_set.get_negative_num())  # 66165 454789
        # 128배치 단위로 뽑아줄때 positive s.s region 32, negative s.s region 96개로 맞춰서 뽑도록 샘플러 설정
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
        # print(len(data_sampler)) # 520832 (num_iter * batch)
        
        # a = list(iter(data_sampler))[:128]
        # print(a)  # [387169, 278321, 19121, .. ], 128개 length
        
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)
        # print(len(data_loader))    # 4069
        # b = next(iter(data_loader))  # 튜플로 묶여있음 : (tensor(B, 3, 227, 227), tensor(0또는 1))
        # print(b[0].shape, b[1].shape) # torch.Size([128, 3, 227, 227]) torch.Size([128])
        
        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()

    return data_loaders, data_sizes


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

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # 모델 저장 규칙 : best validation accuracy 
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    data_loaders, data_sizes = load_data('./data/finetune_car') 

    model = models.alexnet(pretrained=True)
    print(model)
    
    ## [ features layer을 통과한 F.M 출력 ]
    # print(list(model.features.children()))
    # test_model = nn.Sequential(*list(model.features.children()))
    # print(test_model)
    # a = torch.ones(100, 3, 224, 224)
    # out_a = test_model(a)  # torch.Size([100, 256, 6, 6])
    # print(out_a.shape)
    
    ## [ features layer을 통과한 F.M 출력 - 간단버전 ]
    # test_model2 = model.features
    # a = torch.ones(100, 3, 224, 224)
    # out_a = test_model2(a) 
    # print(out_a.shape)
    
    num_features = model.classifier[6].in_features
    # print(num_features)  # 4096
    model.classifier[6] = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Momentum opt + scheduler
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # scheduler epoch 4마다 lr를 줄여줌 (pre lr * gamma)
    # step_size=7 : 7 epochs 마다 lr 변화
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # best 모델 저장
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
