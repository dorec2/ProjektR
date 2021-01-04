import os
import pickle
import numpy as np
from pathlib import Path
import cifarmodels
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import torchvision.transforms as transforms
import math
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt

num_classes = 10
num_labels = 50000
max_count = num_labels // num_classes
img_count = 32
p = 2
batch_size = 32
num_epochs = 50
DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10' / 'cifar-10-batches-py'

transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])


def load_cifar_10():
    def load_cifar_batches(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []

        for fn in filenames:
            with open(os.path.join(DATA_DIR, fn), 'rb') as f:
                data = pickle.load(f, encoding="latin1")
            images.append(np.asarray(data['data'], dtype='float32').reshape(-1, 3, 32, 32) / np.float32(255))
            labels.append(np.asarray(data['labels'], dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
    X_test, y_test = load_cifar_batches('test_batch')

    return X_train, y_train, X_test, y_test


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def apply_transformations(batch: np.ndarray):
    result = torch.zeros(batch.shape)
    for i in range(batch_size):
        result[i] = transform(batch[i])

    return result


def update_w_t(epoch):
    if epoch < num_epochs:
        p = max(0.0, float(epoch) / float(num_epochs))
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0


def evaluate(name, eval_data, eval_labels, eval_model):
    eval_model.eval()
    with torch.no_grad():
        N = len(eval_data)
        cnt_correct = 0
        n_batch_ = N // batch_size
        for batch_ in range(n_batch_):
            batch_train_data_ = torch.FloatTensor(eval_data[batch_ * batch_size:(batch_ + 1) * batch_size, :]).to(device)
            batch_labels_ = torch.LongTensor(eval_labels[batch_ * batch_size:(batch_ + 1) * batch_size]).to(device)

            output = eval_model(batch_train_data_)
            predicted_labels = torch.argmax(output, dim=1)

            cnt_correct += (batch_labels_ == predicted_labels).sum().item()

        accuracy = cnt_correct / N * 100
        print(name + "accuracy = %.2f" % accuracy)
        print()
        return accuracy


X_train, y_train, X_test, y_test = load_cifar_10()
X_train_torch = torch.tensor(X_train)

mask_train = np.zeros(len(y_train), dtype=np.float32)
count = [0] * num_classes
for i in range(len(y_train)):
    label = y_train[i]
    if count[label] > max_count:
        mask_train[i] = 1.0
    count[label] += 1

all_labels = np.array(y_train)
test_labels = np.array(y_test)
y_train[mask_train.astype('bool')] = -1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = cifarmodels.ConvolutionalModel((32, 32), 16, 32, 3, 256, 10)
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

criterion1 = torch.nn.CrossEntropyLoss(ignore_index=-1)
criterion2 = torch.nn.MSELoss()

n_batch = len(X_train) // batch_size

evaluate("Before training ", X_train, all_labels, model)

for epoch in range(num_epochs):
    model.train()
    train_data, labels = shuffle_data(X_train, y_train)
    train_data = torch.FloatTensor(train_data)
    labels = torch.LongTensor(labels)
    for batch in range(n_batch):
        optimizer.zero_grad()

        batch_train_data = torch.FloatTensor(train_data[batch * batch_size:(batch + 1) * batch_size, :])
        batch_labels = torch.LongTensor(labels[batch * batch_size:(batch + 1) * batch_size]).to(device)
        batch_labels.to(device)

        batch_train_data1 = apply_transformations(batch_train_data).to(device)
        batch_train_data2 = apply_transformations(batch_train_data).to(device)

        output1 = model(batch_train_data1)
        output2 = model(batch_train_data2)

        w_t = update_w_t(epoch)

        loss1 = criterion1(output1, batch_labels)
        loss2 = w_t * criterion2(output1, output2) / num_classes
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print("epoch: {}, step: {}/{}, batch_loss: {}"
                  .format(epoch, batch, n_batch, loss.item()))

    scheduler.step()
    train_acc = evaluate("Train ", X_train, all_labels, model)
    evaluate("Test ", X_test, test_labels, model)

evaluate("Test ", X_test, test_labels, model)
