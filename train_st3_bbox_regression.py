import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet

from libs.custom_bbox_regression_dataset import BBoxRegressionDataset
import libs.util as util


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 이미지(bbox crop이미지)와 ti(t_x, t_y, t_w, t_h) 데이터셋 생성
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

    return data_loader


def train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    model.train()  # Set model to training mode
    loss_list = list()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        # Iterate over data.
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # 이미지를 feature extractor에 태워 feature map 얻는다.
            features = feature_model.features(inputs)
            features = torch.flatten(features, 1)
            print(features.shape)  # (128, 9216)  

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # F.E로부터 얻은 F.M을 flatten하여 linear regression에 입력
            # output은 d(p)i가 됨.
            outputs = model(features)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            lr_scheduler.step()

        epoch_loss = running_loss / data_loader.dataset.__len__()
        loss_list.append(epoch_loss)

        print('{} Loss: {:.4f}'.format(epoch, epoch_loss))

        # epoch마다 저장
        util.save_model(model, './models/bbox_regression_%d.pth' % epoch)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return loss_list


def get_model(device=None):
    # classifier 모델 불러오기
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    # 가중치 업데이트 안되게 막기
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


if __name__ == '__main__':
    data_loader = load_data('./data/bbox_regression')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_model = get_model(device)
    # print(feature_model)
    
    """ 
    # output feature map shape 확인 방법
    test_model = nn.Sequential(*list(feature_model.features), feature_model.avgpool)
    print(test_model)
    a = torch.ones(100, 3, 224, 224).to(device)
    out_a = test_model(a)  # torch.Size([100, 256, 6, 6])
    print(out_a.shape)
    exit()
    """
    
    # AlexNet의 마지막 풀링레이어의 shape
    in_features = 256 * 6 * 6
    out_features = 4 # 예측하고픈 bbox coordinates
    model = nn.Linear(in_features, out_features)
    model.to(device)

    criterion = nn.MSELoss()
    # weight_decay : L2 규제 (squares)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    loss_list = train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device,
                            num_epochs=12)
    util.plot_loss(loss_list)
