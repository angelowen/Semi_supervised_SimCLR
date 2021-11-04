import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from data_aug.contrastive_learning_dataset import PlantFinetuneDataset
import torch.utils.data as data
import pandas as pd


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', metavar='DIR', default='./datasets/train',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=['resnet18','resnet50'])
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--pt', default='runs/Nov03_21-03-56_Umaru-203/checkpoint_0600.pth.tar', type=str,
                        help='path to trained model')
    parser.add_argument('--save-path', default='./Finetune_Result', type=str,
                        help='finetune result')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataset
    train_set = PlantFinetuneDataset(args.data,transform=data_transform)
    num_classes = train_set.num_classes
    print("num_classes: ",num_classes)
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    if args.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device)
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes).to(device)

    
    checkpoint = torch.load(args.pt, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):

        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]


    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']


    # freeze all layers but the last fc
    for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2   # fc.weight, fc.bias


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    best = 0.0 
    accuracy_list = {'train': [], 'valid': []}
    epochs = args.epochs
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        accuracy_list['train'].append(top1_train_accuracy.item())
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        accuracy_list['valid'].append(top1_accuracy.item())

        if  top1_accuracy.item() > best:
            best = top1_accuracy.item()
            torch.save(model, os.path.join(args.save_path, './model_weights.pth'))
        print(f"Epoch {epoch+1}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Valid accuracy: {top1_accuracy.item()}\tTop5 Valid acc: {top5_accuracy.item()}")
        
        # plot the loss curve for training and validation
        pd.DataFrame({
            "train-acc": accuracy_list['train'],
            "valid-acc": accuracy_list['valid']
        }).plot()
        plt.xlabel("Epoch"), plt.ylabel("Acc")
        plt.savefig(os.path.join(args.save_path, "Acc_curve.jpg"))
        plt.close()





