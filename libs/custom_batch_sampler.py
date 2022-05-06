import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from libs.custom_finetune_dataset import CustomFinetuneDataset

# ## __main__ 실행시, 경로 수정한 라이브러리 활성화 필요.
# from custom_finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        2진 분류 데이터샘플러
        num_positive: 전체 양성샘플(selective search bbox region) 수
        num_negative: 전체 음성샘플(selective search bbox region) 수
        batch_positive: 양성 샘플의 배치 단위, 32
        batch_negative: 음성 샘플의 배치 단위, 96
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))

        self.batch = batch_negative + batch_positive  # 128 (배치단위: 양성배치 + 음성배치)
        self.num_iter = length // self.batch # 전체 iter수 (전체 bbox region / 배치단위)
        print(f'num iter {self.num_iter}')  # 4069

    def __iter__(self):
        """_summary_
        샘플id를 iteration으로 반환
        """
        sampler_list = list()
        for i in range(self.num_iter): # 전체 iter만큼 반복
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),  # positive bbox에서 32개의 id 뽑는다.(복원 추출), (32,)
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))  # negative bbox에서 96개의 id 뽑는다. (96,)
            )
            # print(tmp.shape) # (128,)
            random.shuffle(tmp)  # 섞는다.
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        # 전체 s.s bbox 갯수를 return
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


def test():
    root_dir = '../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())

    first_idx_list = list(train_sampler.__iter__())[:128]
    print(first_idx_list)
    # positive 배치 갯수
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))


def test2():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)
    data_loader = DataLoader(train_data_set, batch_size=128, sampler=train_sampler, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    test()
