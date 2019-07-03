import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from utils import neg_2d_gaussian_likelihood, transform_and_rotate, \
    build_occupancy_maps, build_humans
from dataset import Dataset


class Trainer(object):
    def __init__(self, model, train_data, val_data, config):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.criterion = neg_2d_gaussian_likelihood
        self.train_data = Dataset(train_data)
        self.val_data = Dataset(val_data)
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=config.get('learning_rate', 0.001),
                                    weight_decay=config.get('weight_decay', 0))
        self.config = config
        #TODO Make batch size configurable
        self.train_loader = DataLoader(
            self.train_data, batch_size=4, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=4, shuffle=True, num_workers=4
        )

    def run(self):
        # print('in run')
        num_epochs = self.config.get('num_epochs', 100)
        print_every = self.config.get('print_every', 10)
        log_path = self.config.get('log_path', 'log')
        running_train_avg_loss = 0
        running_train_avg_l2_error = 0
        c = 0
        f = open(log_path + '/loss.txt', 'w')
        f.write(
            'train running loss, train running l2 error, val loss, val l2 error\n')
        start_time = time()
        # print('finished init parts ')
        for epoch in range(1, num_epochs + 1):
            avg_train_loss, avg_train_l2_error = self.train_one_epoch()
            c += 1
            running_train_avg_loss += avg_train_loss
            running_train_avg_l2_error += avg_train_l2_error
            # print('finished for loop part')
            if epoch % print_every == 0:
                end_time = time()
                val_avg_loss, val_avg_l2_error = self.eval_one_epoch()
                running_train_avg_loss /= c
                running_train_avg_l2_error /= c
                print(
                    'Epoch %d, train loss: %.4f, train l2 error: %.4f, val loss: %.4f, val l2 error: %.4f, time spent: %.4f' % (
                        epoch,
                        running_train_avg_loss,
                        running_train_avg_l2_error,
                        val_avg_loss,
                        val_avg_l2_error,
                        (end_time - start_time)))
                start_time = time()
                f.write("%.4f, %.4f, %.4f, %.4f\n" % (
                running_train_avg_loss, running_train_avg_l2_error,
                val_avg_loss, val_avg_l2_error))
                f.flush()
                c, running_train_avg_loss, running_train_avg_l2_error = 0, 0, 0
        f.close()

    def train_one_epoch(self):
        self.model.train()  # sets the model to train mode

        train_loss = 0
        train_l2_error = 0
        c = 0

        for batch, labels in self.train_loader:
            pred_t = self.model(batch)
            loss, l2_error = self.criterion(pred_t, labels)
            c += 1
            train_loss += loss.item()
            train_l2_error += l2_error.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_loss/c, train_l2_error/c

    def eval_one_epoch(self):
        self.model.eval()   # Put model in eval mode
        val_loss = 0
        val_l2_error = 0
        c = 0
        for batch, labels in self.val_loader:
            with torch.no_grad():   # We won't back-propagate based on eval
                pred_t = self.model(batch)
                loss, l2_error = self.criterion(pred_t, labels)
                c += 1
                val_loss += loss.item()
                val_l2_error += l2_error.item()
        return val_loss / c, val_l2_error / c
