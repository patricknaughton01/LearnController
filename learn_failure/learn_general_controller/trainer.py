import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from utils import neg_2d_gaussian_likelihood, transform_and_rotate, \
    build_occupancy_maps, build_humans

class Trainer(object):
    def __init__(self, model, train_loader, val_loader, config):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.criterion = neg_2d_gaussian_likelihood
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(model.parameters(), 
                                    lr=config.get('learning_rate', 0.005),
                                    weight_decay=config.get('weight_decay', 0))
        self.config = config
    
    def run(self):
        # print('in run')
        num_epochs = self.config.get('num_epochs', 100)
        print_every = self.config.get('print_every', 10)
        log_path = self.config.get('log_path', 'log')
        running_train_avg_loss = 0
        running_train_avg_l2_error = 0
        c = 0
        f = open(log_path + '/loss.txt', 'w')
        f.write('train running loss, train running l2 error, val loss, val l2 error\n')
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
                print('Epoch %d, train loss: %.4f, train l2 error: %.4f, val loss: %.4f, val l2 error: %.4f, time spent: %.4f' % (
                                                                  epoch,
                                                                  running_train_avg_loss, 
                                                                  running_train_avg_l2_error,
                                                                  val_avg_loss,
                                                                  val_avg_l2_error,
                                                                  (end_time - start_time)))
                start_time = time()
                f.write("%.4f, %.4f, %.4f, %.4f\n" % (running_train_avg_loss, running_train_avg_l2_error, val_avg_loss, val_avg_l2_error))
                f.flush()
                c, running_train_avg_loss, running_train_avg_l2_error = 0, 0, 0   
        f.close()             

    def train_one_epoch(self):
        self.model.train() #sets the model to train mode

        train_loss = 0
        train_l2_error = 0
        c = 0
        
        # print('finished init parts 2')
        # train_loader is the loaded training data
        # batch_states, batch_seq_lengths, batch_targets
        for states, seq_lengths, targets, _ in self.train_loader():
            
            # print(states.shape)
            # print(seq_lengths)
            # print(targets.shape)
            # print(_)

            seq_lengths = torch.from_numpy(seq_lengths).long()
            targets = torch.from_numpy(targets).float()

            # states: seq_len x batch_size x dim
            seq_len = states.shape[0]
            # print(states.shape)
            outputs = []
            h_t = None

            # flag_new_pred = 0
            # s = time()
            # print('second for loop')
            for i in range(seq_len):
                cur_states = states[i]

                # if flag_new_pred is 1:
                    
                    # cur_states[:, 0:2] = (new_pred.data).cpu().numpy() # (Variable(x).data).cpu().numpy()
                    
                    # print("cur_states in for loop", cur_states)
                cur_rotated_states = transform_and_rotate(cur_states)
                # now state_t is of size: batch_size x num_human x dim
                batch_size = cur_states.shape[0]

                batch_occupancy_map = []

                for b in range(batch_size):
                    occupancy_map = build_occupancy_maps(build_humans(cur_states[b]))
                    batch_occupancy_map.append(occupancy_map)

                batch_occupancy_map = torch.stack(batch_occupancy_map)#[:,
                # 1:, :]
                state_t = torch.cat([cur_rotated_states, batch_occupancy_map], dim=-1)

                #############################################
                # this function calls forward from model.py #
                #############################################

                pred_t, h_t = self.model(state_t, h_t)

                # new_pred = torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2]
                # flag_new_pred = 1
                # print('pred_t')
                # print(pred_t)
                # print('done iwith for loops')
                outputs.append(pred_t)
            # e = time()
            # print('one batch spent %.4f seconds' % (e-s))
            outputs = torch.stack(outputs)
            self.optimizer.zero_grad()
            
            loss, l2_error = self.criterion(outputs, targets, seq_lengths)
            train_loss += loss.item()
            train_l2_error += l2_error.item()
            c += 1
            loss.backward()
            self.optimizer.step()

            # print('all done')

        return train_loss / c, train_l2_error / c

    def eval_one_epoch(self):
        if self.val_loader is None:
            return 0, 0
        self.model.eval()
        val_loss = 0
        val_l2_error = 0
        c = 0
        for states, seq_lengths, targets, _ in self.val_loader():
            seq_lengths = torch.from_numpy(seq_lengths).long()
            targets = torch.from_numpy(targets).float()

            # states: seq_len x batch_size x dim
            seq_len = states.shape[0]
            outputs = []
            h_t = None
            for i in range(seq_len):
                cur_states = states[i]
                cur_rotated_states = transform_and_rotate(cur_states)
                # now state_t is of size: batch_size x num_human x dim
                batch_size = cur_states.shape[0]
                batch_occupancy_map = []
                for b in range(batch_size):
                    occupancy_map = build_occupancy_maps(build_humans(cur_states[b]))
                    batch_occupancy_map.append(occupancy_map)
                batch_occupancy_map = torch.stack(batch_occupancy_map)#[:,
                # 1:, :]
                state_t = torch.cat([cur_rotated_states, batch_occupancy_map], dim=-1)
                pred_t, h_t = self.model(state_t, h_t)
                outputs.append(pred_t)
            outputs = torch.stack(outputs)
            
            loss, l2_error = self.criterion(outputs, targets, seq_lengths)
            val_loss += loss.item()
            val_l2_error += l2_error.item()
            c += 1

        return val_loss / c, val_l2_error / c
