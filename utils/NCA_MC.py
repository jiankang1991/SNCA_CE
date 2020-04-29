
import torch
from torch import nn
from torch.autograd import Function
import math


eps = 1e-8



class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out



class NCACrossEntropy_MoCo(nn.Module):
    """Fixed-size queue with momentum encoder
    self.queueSize equals to self.outputSize
    """
    def __init__(self, inputSize, outputSize, K, labels, T=0.07, margin=0):
        super().__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin
        # outputSize: the total number of images
        self.outputSize = outputSize
        # inputSize: dim
        self.inputSize = inputSize
        # queue size
        self.queueSize = K
        self.T = T
        self.index = 0

        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, x, x_hat, indexes):

        batchSize = x.shape[0]

        # inner product
        sim_x_m = torch.mm(x.data, self.memory.t())
        sim_x_m.div_(self.T) # batchSize * SampleNum

        sim_x_m_exp = torch.exp(sim_x_m)

        # labels for currect batch
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, self.queueSize).eq_(self.labels)

        # self prob exclusion, hack with memory for effeciency
        sim_x_m_exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(sim_x_m_exp, same.float()).sum(dim=1)
        Z = sim_x_m_exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)


        # # update memory
        with torch.no_grad():
            # out_ids = torch.arange(batchSize).cuda()
            # out_ids += self.index
            # out_ids = torch.fmod(out_ids, self.queueSize)
            # out_ids = out_ids.long()
            self.memory.index_copy_(0, indexes.data.view(-1), x_hat.data)
            # self.index = (self.index + batchSize) % self.queueSize


        return - loss / batchSize






















