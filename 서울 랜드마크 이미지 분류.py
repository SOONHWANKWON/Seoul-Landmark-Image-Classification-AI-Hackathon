# 필요한 라이브러리 임포트

import pandas as pd
import os
from glob import glob
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoade
import matplotlib.pyplot as pltr, Dataset
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

label_df = pd.read_csv("D:/dataset/train.csv")
label_df

# pandas의 read_csv 메서드를 사용하여 CSV 파일로부터 데이터를 로드

label_df = pd.read_csv("D:/dataset/train.csv")

def get_train_data(data_dir):
    img_path_list = []
    label_list = []
    
    img_path_list.extend(glob(os.path.join(data_dir, '*.PNG')))
    img_path_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        
    label_list.extend(label_df['label'])
                
    return img_path_list, label_list

def get_test_data(data_dir):
    img_path_list = []
    
    img_path_list.extend(glob(os.path.join(data_dir, '*.PNG')))
    img_path_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    return img_path_list

#데이터의 요약 정보를 확인

label_df.info()

# 학습 이미지 경로와 라벨, 테스트 이미지 경로 불러오기

all_img_path, all_label = get_train_data("D:/dataset/train")
test_img_path = get_test_data("D:/dataset/test")

all_img_path[:5]

test_img_path[:5]

# gpu설정

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda')

if torch.cuda.is_available():    
    #device = torch.device("cuda:0")
    print('Device:', device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))


# 하이퍼 파라미터 값을 미리 지정

CFG = {'IMG_SIZE':128,
       'EPOCHS':50,
       'LEARNING_RATE':2e-2,
       'BATCH_SIZE':12,
       'SEED':41}

# 무작위성을 제어하기 위해 andom seed를 고정

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# 데이터 전처리 작업

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index): 
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image
    
    def __len__(self): 
        return len(self.img_path_list)

# 이미지 추출해 확인해보기

testdataset = CustomDataset(all_img_path, all_label, train_mode=False)
plt.imshow(testdataset.__getitem__(0))

# 전체 데이터셋을 훈련용 데이터셋과 검증용 데이터셋으로 나누기

train_len = int(len(all_img_path)*0.75)
Vali_len = int(len(all_img_path)*0.25)

train_img_path = all_img_path[:train_len]
train_label = all_label[:train_len]

vali_img_path = all_img_path[train_len:]
vali_label = all_label[train_len:]

# 이미지 전처리 작업

train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 처음은 train dataset 생성하기
train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform)
# 그다음 DataLoader 사용해 batch만들기
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=test_transform)
vali_loader = DataLoader(vali_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

train_batches = len(train_loader)
vali_batches = len(vali_loader)

print('total train imgs :',train_len,'/ total train batches :', train_batches)
print('total valid imgs :',Vali_len, '/ total valid batches :', vali_batches)

# 모델링 작업

from torchvision import models
from torchvision.models import efficientnet_b7
model = models.efficientnet_b7(pretrained=True)

model.classifier

# 모델 학습을 위한 매개변수를 정의

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = None

# 모델을 학습시키고 평가

from tqdm import tqdm

def train(model, optimizer, train_loader, scheduler, device, criterion, vali_loader):
    model.to(device)
    n = len(train_loader)
    best_acc = 0

    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train()
        running_loss=0.0

        # 미니 배치(batch) 단위로 데이터를 처리하고 가중치를 업데이트
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            logit = model(img)
            loss = criterion(logit, label)

            #역전파
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))

        if scheduler is not None:
            scheduler.step()

        # Validation set 평가 
        model.eval()  
        vali_loss = 0.0
        correct = 0
        with torch.no_grad():
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item() # 예측값과 실제값이 맞으면 1 아니면 0으로 합산

        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vali_loader), correct, len(vali_loader.dataset), 100 * correct / len(vali_loader.dataset)))

        # 최적 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            
            # 디렉토리 생성 확인 및 모델 저장
            save_path = 'D:/dataset/best_model.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            print('Model Saved.')

train(model, optimizer, train_loader, scheduler, device, criterion, vali_loader)

# 학습된 best_model을 가지고 test 셋의 라벨을 추론

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred

test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

checkpoint = torch.load('D:/dataset/best_model.pth')
model.load_state_dict(checkpoint)

preds = predict(model, test_loader, device)

# SCV파일로 저장

submission = pd.read_csv('D:/dataset/sample_submission.csv')

submission['label'] = preds

submission.to_csv('D:/dataset/submission2.csv', index=False)
