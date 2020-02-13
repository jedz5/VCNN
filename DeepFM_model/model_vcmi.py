# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DeepFM_model.process_vcmi_data import fineTune_Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
class DeepFM_vcmi(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes=[140, 24, 8, 8, 2, 2, 2, 2, 2, 2], embedding_size=[10,4,4,4,4,4,4,4,4,4],slot_emb_size = 16,
                 hidden_dims=[1024, 1024], num_slots=7, dropout=[0.5, 0.5,0.5],
                 use_cuda=True, verbose=False,log_path = "hello"):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.log_path = log_path
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.slot_emb_size = slot_emb_size
        self.hidden_dims = hidden_dims
        self.num_slots_out = num_slots
        self.dtype = torch.long
        # self.bias = torch.nn.Parameter(torch.randn(1))
        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("using device {}".format(self.device))
        """
            init fm part
        """
        # self.fm_first_order_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.creature_prop_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size[idx],padding_idx=0) for idx,feature_size in enumerate(self.feature_sizes)])
        self.creature_slot_embedding = nn.Linear(sum(self.embedding_size),self.slot_emb_size)
        """
            init deep part
        """
        all_dims = [self.slot_emb_size * 14] + \
                   self.hidden_dims + [self.num_slots_out]
        for i in range(1, 1 +len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(dropout[i - 1]))
        # setattr(self, 'batchNorm_kill',
        #         nn.BatchNorm1d(all_dims[-1]))
        setattr(self, 'batchNorm_win',
                nn.BatchNorm1d(1))
        setattr(self,'win_out',
                nn.Linear(all_dims[-2], 1))


    def forward(self, Xi, Xv):
        """
            fm part
        """

        # fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                           enumerate(self.fm_first_order_embeddings)]
        # # fm_first_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i])  for i, emb in enumerate(self.fm_first_order_embeddings)]
        # fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        # fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                            enumerate(self.fm_second_order_embeddings)]
        # # fm_second_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        # fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        # fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
        #                                  fm_sum_second_order_emb  # (x+y)^2
        # fm_second_order_emb_square = [
        #     item * item for item in fm_second_order_emb_arr]
        # fm_second_order_emb_square_sum = sum(
        #     fm_second_order_emb_square)  # x^2+y^2
        # fm_second_order = (fm_sum_second_order_emb_square -
        #                    fm_second_order_emb_square_sum) * 0.5

        creature_prop_embeded = []
        for i in range(14):
            slot = []
            for j, emb in enumerate(self.creature_prop_embeddings):
                slot.append(emb(Xi[:, i, j]) * Xv[:, i, j])
            creature_prop_embeded.append(torch.cat(slot, 1))
        slots = []
        for i in range(14):
            slot_i = self.creature_slot_embedding(creature_prop_embeded[i])
            slot_i = F.relu(slot_i)
            slots.append(slot_i)
        """
            deep part
        """
        deep_emb = torch.cat(slots, 1)
        deep_out = deep_emb
        for i in range(1, 1 + len(self.hidden_dims)):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = F.relu(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        #i+1
        i = 1 + len(self.hidden_dims)
        kill_out = getattr(self, 'linear_' + str(i))(deep_out)
        # kill_out = getattr(self, 'batchNorm_kill')(kill_out)
        win_out = getattr(self, 'win_out')(deep_out)
        win_out = getattr(self, 'batchNorm_win')(win_out)
        win_out = torch.sigmoid(win_out)
        """
            sum
        """
        # total_sum = torch.sum(fm_first_order, 1) + \
        #             torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return win_out,kill_out

    def fit(self, loader_train, loader_val, optimizer, init_lr,weight_decay,lr_decay_per_ep,verbose=False, print_every=50,best_start = 0.1,ep_start = 0,ep_end=20):
        if os.path.exists(self.log_path):
            tf.io.gfile.rmtree("hello")
        log_writer = SummaryWriter(self.log_path)
        model = self.train().to(device=self.device)
        criterion_Kill = nn.MSELoss(reduction='mean')
        criterion_win = nn.BCELoss()
        best = best_start
        state = 0
        print_per_ep = int(len(loader_train) / print_every)
        print("print_per_ep = {}".format(print_per_ep))
        # scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)  # set up scheduler
        for ep in range(ep_start,ep_end):
            print("epoch {}".format(ep))
            lr = 0
            if ep % lr_decay_per_ep == 0:
                lr = init_lr * (weight_decay ** (ep // lr_decay_per_ep))
                if lr < 1E-4:
                    lr = 1E-4
                print('LR = {} weight_decay = {} per {} eps'.format(lr,weight_decay,lr_decay_per_ep))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            for t, (xi, xv, y_ka,y_a,y_k,y_v,y_k_all,k_path) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y_k = y_k.to(device=self.device, dtype=torch.float)
                y_v = y_v.to(device=self.device, dtype=torch.float)
                mask = (y_a != 0).to(device=self.device, dtype=torch.float)
                win,pred = model(xi, xv)
                pred = pred * mask      #.cpu().detach().numpy()
                kill_loss = 50*criterion_Kill(pred,y_k)
                win_loss = criterion_win(win,y_v)
                loss = win_loss + kill_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose and t % print_every == 0:
                    indices_train = np.random.choice(list(range(len(loader_train.dataset))),int(len(loader_train.dataset)*0.2))
                    indices_val = np.random.choice(list(range(len(loader_val.dataset))),int(len(loader_val.dataset) * 0.6))
                    # Creating PT data samplers and loaders:
                    train_sampler = SubsetRandomSampler(indices_train)
                    valid_sampler = SubsetRandomSampler(indices_val)
                    loader_train_acc = DataLoader(loader_train.dataset, batch_size=2000, sampler=train_sampler)
                    loader_val_acc = DataLoader(loader_val.dataset, batch_size=2000, sampler=valid_sampler)
                    # loader_train_acc = loader_train
                    # loader_val_acc = loader_val
                    ii = (print_per_ep * ep + t / print_every)
                    current_w_train, current_k_train = self.check_accuracy(loader_train_acc, model,train_mode=True)
                    print('Iteration %d, win_loss = %.4f, kill_loss = %.4f, all = %.4f' % (ii, win_loss.item(),kill_loss.item(),loss.item()))
                    current_w,current_k = self.check_accuracy(loader_val_acc, model)
                    log_writer.add_scalars('data/wins', {"train_win_error": current_w_train,"test_win_error": current_w}, ii)
                    log_writer.add_scalars('data/kills', {"train_kill_error": current_k_train,"test_kill_error": current_k},ii)
                    log_writer.add_scalar('data/learning_rate',lr,ii)
                    # log_writer.add_scalars('data/wins', {},ii)
                    # log_writer.add_scalars('data/kills', {},ii)

                    # scheduler.step(current_k, ep)  # update lr if needed
                    if current_k < best:
                        best = current_k
                        state = {"net":self.state_dict(),'optimizer': optimizer.state_dict(),'best':best,'epoch': ep}
                        torch.save(state,"net_pram_temp.pkl")
                        print("saved model best = {} {}".format(current_w,current_k))
                        if best < 0.02:
                            print("chabuduole~~~")
                            log_writer.close()
                            return state
        log_writer.close()
        return state
    def check_accuracy(self, loader, model,train_mode=False):
        if train_mode:
            print('---------Checking accuracy on train set----------')
        else:
            print('*********Checking accuracy on test set**********')
        # all_error = 0
        win_error = 0
        kill_error = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y_ka,y_a,y_k,y_v,y_k_all,k_path in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=torch.float)
                y_ka = y_ka.to(device=self.device, dtype=torch.float)
                y_a = y_a.to(device=self.device, dtype=torch.float)
                y_v = y_v.to(device=self.device, dtype=	torch.uint8)
                mask = (y_a!= 0).to(device=self.device, dtype=torch.float)
                win,pred_kill = model(xi, xv)
                pred_kill = torch.clamp(pred_kill, 0, 1.0) * mask
                pred_kill = torch.round(pred_kill * y_a)
                pred_v = (win > 0.5)
                win_wrong = (pred_v != y_v)
                win_error += win_wrong.float().sum()
                each_kill = torch.abs(pred_kill - y_ka) / (y_a + 1E-6)
                each_kill = y_v.float() * each_kill
                each_kill[win_wrong.squeeze()] = 1
                total_kill = each_kill.sum(1, keepdim=True) / mask.sum(1, keepdim=True)
                total_kill[win_wrong] = 1
                kill_error += total_kill.sum()
                num_samples += pred_kill.size(0)
            acc_win = float(win_error) / num_samples
            acc_win = round(acc_win,4)
            acc_kill = float(kill_error) / num_samples
            acc_kill = round(acc_kill, 4)
            print('Got %d sample win_error (%.2f%%). kill_error (%.2f%%)' % (num_samples,100 * acc_win,100 * acc_kill))
            return acc_win,acc_kill



