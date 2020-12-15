import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.utils.data import DataLoader
from DPQ import DPQ
from datasets import DCMQDataSet
from evaluations import i2t, t2i, Reconstruct
from torch.utils.tensorboard import SummaryWriter
import logging


class DCMQSolver():
    def __init__(self, config_file):
        self.config_file = config_file
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_data()
        print('DCMQSolver Constructing successfully!')
    def build_model(self):
        M = self.config_file['M']
        K = self.config_file['K']
        D = self.config_file['D']
        self.model = DPQ(M, K, D).cuda()
    def build_data(self):
        data_path = self.config_file['data_path']
        batch_size = self.config_file['batch_size']
        train_dataset = DCMQDataSet(data_path).Train()
        self.length_train = len(train_dataset)
        self.query_dataset = {'i2t': DCMQDataSet(data_path).Query('i2t'),
                              't2i': DCMQDataSet(data_path).Query('t2i')}
        retrieval_dataset = {'i2t': DCMQDataSet(data_path).Retrival('i2t'),
                                  't2i': DCMQDataSet(data_path).Retrival('t2i')}
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True, num_workers=0)
        # self.query_dataloader = {'i2t': DataLoader(query_dataset['i2t'],
        #                                            batch_size=batch_size, shuffle=True, num_workers=0),
        #                          't2i': DataLoader(query_dataset['t2i'],
        #                                            batch_size=batch_size, shuffle=True, num_workers=0)}
        self.retrieval_dataloader = {'i2t': DataLoader(retrieval_dataset['i2t'],
                                                       batch_size=batch_size, shuffle=False, num_workers=0),
                                     't2i': DataLoader(retrieval_dataset['t2i'],
                                                       batch_size=batch_size, shuffle=False, num_workers=0)}
    def build_optimizer(self):
        lr = self.config_file['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, amsgrad=True)
    def build_scheduler(self):
        lr_decay = self.config_file['lr_decay']
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
    def train(self):
        # tensorboard
        tb_writer = SummaryWriter(self.config_file['tb_dir'])
        self.model.train()
        num_epoch = self.config_file['num_epoch']
        for i in range(num_epoch):
            iter = 0
            for data in self.train_dataloader:
                iter += 1
                input = data.cuda().float() * 100
                self.optimizer.zero_grad()
                _, loss, soft_loss, hard_loss, joint_loss = self.model(input)
                loss.backward()
                self.optimizer.step()
                total_iter = i * round(self.length_train / self.config_file['batch_size']) + iter
                tb_writer.add_scalar('loss', loss, total_iter)
                if iter % 100 == 0:
                    print(f"Epoch {i} / Iter {iter} with Loss: {loss.detach().mean()}")
                    print(f"Soft Loss {soft_loss}, hard Loss {hard_loss}, joint Loss {joint_loss}")
                if iter % 1000 == 0:
                    self.validate(tb_writer, total_iter)
            self.scheduler.step()
    def encode(self):
        i2tcodes = list()
        t2icodes = list()
        # i2torigins = list()
        # t2iorigins = list()
        i2tlosses = list()
        t2ilosses = list()
        for data in self.retrieval_dataloader['i2t']:
            input = data.cuda().float() * 100
            i2tcode, _, _, i2tloss, _ = self.model(input)
            i2tcodes.append(i2tcode)
            # i2torigins.append(input)
            i2tlosses.append(i2tloss.detach().mean())
        i2tlosses = torch.tensor(i2tlosses).mean()
        # i2torigins = torch.cat(i2torigins, dim=0)
        i2tcodes = torch.cat(i2tcodes, dim=0)
        for data in self.retrieval_dataloader['t2i']:
            input = data.cuda().float() * 100
            t2icode, _, _, t2iloss, _ = self.model(input)
            t2icodes.append(t2icode)
            # t2iorigins.append(input)
            t2ilosses.append(t2iloss.detach().mean())
        t2ilosses = torch.tensor(t2ilosses).mean()
        # t2iorigins = torch.cat(t2iorigins, dim=0)
        t2icodes = torch.cat(t2icodes, dim=0)
        # return i2tcodes, t2icodes, i2torigins, t2iorigins, i2tlosses, t2ilosses
        return i2tcodes, t2icodes, i2tlosses, t2ilosses



    def validate(self, tb_writer, total_iter):
        self.model.eval()
        with torch.no_grad():
            # get retrieval codes
            # i2tcodes, t2icodes, i2torigins, t2iorigins, i2tloss, t2iloss = self.encode()
            i2tcodes, t2icodes, i2tloss, t2iloss = self.encode()
            quantized_cap = Reconstruct(self.model.Codebook().cuda(), i2tcodes)
            quantized_img = Reconstruct(self.model.Codebook().cuda(), t2icodes)
            query_img = self.query_dataset['i2t'].data * 100
            query_txt = self.query_dataset['t2i'].data * 100
            # caption retrieval
            # (r1, r5, r10, medr, meanr) = i2t(query_img, i2torigins)
            # print("Original Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
            #       (r1, r5, r10, medr, meanr))
            (r1, r5, r10, medr, meanr) = i2t(query_img, quantized_cap)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                         (r1, r5, r10, medr, meanr))
            # image retrieval
            # (r1, r5, r10, medr, meanr) = t2i(query_txt, t2iorigins)
            # print("Original Text to image %.1f, %.1f, %.1f, %.1f, %.1f" %
            #       (r1, r5, r10, medr, meanr))
            (r1i, r5i, r10i, medri, meanr) = t2i(query_txt, quantized_img)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                         (r1i, r5i, r10i, medri, meanr))
            print("Reconstruct Loss by model:i2t %.4f, t2i %.4f"%(i2tloss, t2iloss))
            reconstruct_img = (query_img.cuda().float() - quantized_img) ** 2
            reconstruct_img = reconstruct_img.mean()
            reconstruct_txt = (query_txt.cuda().float() - quantized_cap) ** 2
            reconstruct_txt = reconstruct_txt.mean()
            print("Reconstruct Loss by cal:i2t %.4f, t2i %.4f" % (reconstruct_img, reconstruct_txt))
            tags = ['t2iR@1', 't2iR@5', 't2iR@10', 'i2tR@1', 'i2tR@5', 'i2tR@10']
            tb_writer.add_scalar(tags[0], r1i, total_iter)
            tb_writer.add_scalar(tags[1], r5i, total_iter)
            tb_writer.add_scalar(tags[2], r10i, total_iter)
            tb_writer.add_scalar(tags[3], r1, total_iter)
            tb_writer.add_scalar(tags[4], r5, total_iter)
            tb_writer.add_scalar(tags[5], r10, total_iter)
        self.model.train()


if __name__ == '__main__':
    config_file = {'lr': 1e-1,
                   'lr_decay': 0.99,
                   'data_path': '/home/zhujinkuan/CVSE/', #'/mnt/hdd1/zhujinkuan/cvse_coco/',
                   'tb_dir': '/home/zhujinkuan/CrossModalQuant/logs/',
                   'batch_size': 500,
                   'num_epoch': 100,
                   'M': 2,
                   'K': 256,
                   'D': 1024
                  }
    solver = DCMQSolver(config_file)
    solver.train()
    solver.validate()


    
