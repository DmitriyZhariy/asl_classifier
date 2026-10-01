import copy
import json
import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from asl import convert_model as cm

TRAIN_PATH = Path(os.getenv("TRAIN_PATH", "datasets/train"))
TEST_PATH = Path(os.getenv("TEST_PATH", "datasets/test"))
FORMAT_PATH = Path(os.getenv("FORMAT_PATH", "datasets/format.json"))

# TODO: ЗАМЕНИТЬ
model_path = 'artifacts'

MODEL_NAME = os.getenv("MODEL_NAME", "asl")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "asl")
MIN_GAIN = float(os.getenv("GATE_MIN_GAIN", "0.0"))
SEED = 42
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList"]

# ONNX_PATH = Path("artifacts/shufflenet_v1.onnx")

def init_training() -> torch.device:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    with open(FORMAT_PATH) as file:
        classes = json.load(file)
    classes = list(classes.values())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device, classes

class EarlyStopping:
    def __init__(self, patience=5, save_best=True, filepath=None, metric='accuracy'):
        self.patience = patience
        self.save_best = save_best
        self.filepath = filepath
        self.metric=metric

        self.counter = 0
        self.best_val = None
        self.best_model = None
        self.early_stop = False

        if self.metric == 'accuracy':
            self.best_val = -float('inf')
            self.compare = lambda a, b: a > b
        elif self.metric == 'loss':
            self.best_val = float('inf')
            self.compare = lambda a, b: a < b

    def step(self, model, cur_metric_value, epoch):
        if self.compare(cur_metric_value, self.best_val):
            self.best_val = cur_metric_value
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: no improvement for {self.counter} epochs")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.save_best:
                torch.save(self.best_model, 
                        self.filepath.format(metric=self.metric, metric_val=self.best_val))
            else:
                torch.save(model.state_dict(), 
                        self.filepath.format(metric=self.metric, metric_val=cur_metric_value))
            print(f"EarlyStopping: stopping training at epoch {epoch + 1}")

class AlbumentationsTransform:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.albumentations_transform(image=img)
        return augmented['image']

def loss_model(device, model, loss_func, data):
  model.eval()
  Q = 0
  count = 0

  for x, y in data:
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
      p = model(x)
      loss = loss_func(p, y)
      Q += loss.item()
      count += 1

  Q /= count

  return Q


def accuracy_model(model, device, data):
  model.eval()
  Q = 0

  for x, y in data:
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
      p = model(x)
      p = torch.argmax(p, dim=1)
      Q += torch.sum(p == y).item()

  count = len(data.dataset)

  Q /= count
  
  return Q

def split_data(transform, train_size=0.8, batch_size=100, album=False):
    if album:
        transform = AlbumentationsTransform(transform)
    dataset_train = ImageFolder(TRAIN_PATH, transform=transform)

    train_size = int(train_size * len(dataset_train))
    val_size = len(dataset_train) - train_size
    d_train, d_val = data.random_split(dataset_train, [train_size, val_size])

    train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
    train_data_val = data.DataLoader(d_val, batch_size=batch_size, shuffle=False)

    d_test = ImageFolder(TEST_PATH, transform=transform)
    test_data = data.DataLoader(d_test, batch_size=batch_size, shuffle=False)

    return train_data, train_data_val, test_data

def get_submodule_names(module, max_depth=2, prefix=''):
    # вывод списка названий суб-блоков НН 
    if max_depth == 0:
        return []
    
    names = []
    for name, child in module.named_children():
        if (name == 'fc') or ('pool' in name):
            continue
        
        full_name = f"{prefix}{name}" if prefix == '' else f"{prefix}.{name}"
        
        # Проверяем, есть ли у ребенка trainable параметры
        children = list(child.named_children())
        if max_depth > 1 and children:
            # Есть дети — рекурсивно ищем в них
            names.extend(get_submodule_names(child, max_depth - 1, full_name))
        else:
            # Дети есть, но max_depth исчерпан, или детей нет — добавляем этот модуль
            names.append(full_name)

    return names


def get_submodule_by_name(model, submodule_name):
    # вывод блока НН по его названию
    parts = submodule_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part, None)
    return module

# Функция разморозки блока
def unfreeze_block(model, block_name):
    block = get_submodule_by_name(model, block_name)
    if block is None:
        return False

    # Проверка, разморожен ли уже блок
    any_frozen = any(not param.requires_grad for param in block.parameters())
    if not any_frozen:
        return False

    for param in block.parameters():
        param.requires_grad = True
    return True  


# Функция размораживания следующих блоков по номеру эпохи
def gradual_unfreeze(model, epoch, freeze_order, unfreeze_step=5):
    idx = epoch // unfreeze_step - 1
    if 0 <= idx < len(freeze_order):
        # вывод при успешной разморозке
        if unfreeze_block(model, freeze_order[idx]):
            print(f'Unfroze block: {freeze_order[idx]} at epoch {epoch}')

def update_scheduler(lr_scheduler, val_res):
    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_res)
        else:
            lr_scheduler.step()

def do_callbacks(model, callbacks, epoch, val_res):
    for cb in callbacks:
        # фиксируем чекпоинты
        # if isinstance(cb, ModelCheckpoint):
        #     cb.step(model, val_res, epoch)

        # проверяем на улучшение
        if isinstance(cb, EarlyStopping):
            cb.step(model, val_res, epoch)

            if cb.early_stop:
                if cb.save_best:
                    model.load_state_dict(cb.best_model)
                return True
    return False


def train_model_with_callbacks_finetune(
    model, device, 
    train_data, train_data_val,
    loss_func, epochs,
    freeze_order, unfreeze_step=5,
    opt_lr=0.001, lr_sch_step=5, lr_sch_gamma=0.1,
    metric='accuracy',
    callbacks=None,
):
    model.to(device)
    res_lst_val = []
    res_lst = []

    for epoch in range(epochs):
        model.train()
        loss_mean = 0
        lm_count = 0
        train_tqdm = tqdm(train_data, leave=False)

        gradual_unfreeze(model, epoch, freeze_order, unfreeze_step)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_sch_step, gamma=lr_sch_gamma)

        for x_train, y_train in train_tqdm:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            predict = model(x_train)
            loss = loss_func(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = loss.item() / lm_count + loss_mean * (1 - 1 / lm_count)
            train_tqdm.set_description(f"Epoch [{epoch+1}/{epochs}], loss: {loss_mean:.4f}")

        # метрики на обучении и валидации
        if metric == 'accuracy':
            train_res = accuracy_model(model, device, train_data)
            val_res = accuracy_model(model, device, train_data_val)
        elif metric == 'loss':
            train_res = loss_mean
            val_res = loss_model(device, model, loss_func, train_data_val)

        res_lst.append(train_res)
        res_lst_val.append(val_res)

        print(f'Epoch [{epoch+1}/{epochs}] | {metric}_train={train_res:.3f}, {metric}_val={val_res:.3f}')

        # callbacks
        early_stop = do_callbacks(model, callbacks, epoch, val_res)
        if early_stop:
            return model, res_lst, res_lst_val
        
        # scheduler
        update_scheduler(lr_scheduler, val_res)
    return model, res_lst, res_lst_val


def main():
    device, classes = init_training()
    
    weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
    model = torchvision.models.shufflenet_v2_x1_5(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False

    # Разморозить fc слой
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes)).to(device)
    for param in model.fc.parameters():
        param.requires_grad = True

    loss_func = nn.CrossEntropyLoss()
    epochs = 2

    freeze_order = get_submodule_names(model, max_depth=1)[::-1]

    transforms = A.Compose(
        [
            A.Affine(
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.OpticalDistortion(p=0.2),
                A.GaussNoise(p=0.2)
            ], p=1),
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_size = 0.8
    batch_size = 100
    album = True

    train_data, train_data_val, test_data = split_data(transforms, train_size=train_size, 
                                                   batch_size=batch_size, album=album)

    # chck_p_path = os.path.join(model_path, 'model_epoch-{epoch}_{metric}-{metric_val:.4f}.pt')
    erly_st_path = os.path.join(model_path, 'best_model_{metric}-{metric_val:.4f}.pt')

    metric = 'accuracy'

    earlystop_cb = EarlyStopping(
        filepath=erly_st_path, 
        patience=5, 
        save_best=True,
        metric=metric
    )

    model, res_lst, res_lst_val = train_model_with_callbacks_finetune(
        model, device, train_data, train_data_val,
        loss_func, epochs,
        freeze_order, unfreeze_step=5,
        opt_lr=0.001, lr_sch_step=2, lr_sch_gamma=0.1,
        metric=metric,
        callbacks=[earlystop_cb]
    )

    acc = accuracy_model(model, device, test_data)

    onnx_path = Path(f'{model_path}/best_model_{metric}-{acc:.4f}.onnx')

    onnx_model = cm.convert_pt_to_onnx(model, onnx_path)


if __name__ == '__main__':
    main()