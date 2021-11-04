import logging
import os
import sys

import torch
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.INFO)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        
        features = F.normalize(features, dim=1)
        labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)


        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        best = 0.0
        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        for epoch_counter in range(self.args.epochs):
            for images in tqdm(train_loader):

                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                # with autocast(enabled=self.args.fp16_precision):
                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.info(f"Epoch: {epoch_counter+1}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if top1[0] > best:
                best = top1[0]
                # save model checkpoints
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        logging.info("Training has finished.")
        
