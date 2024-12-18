import torch
import torch.nn.functional as F
import random
import numpy as np



def create_syn_neg(positive, negative, s, mix_ratio):
    # During training, we create synthetic negatives with mixing random negatives and a random positive sample

         harder_synthetic = torch.tensor([], device=negative.device)
         negatives = torch.tensor([], device=negative.device)
         positives = torch.tensor([], device=negative.device)
         positive = positive.transpose(1,0)
         for i in range(s):
            index1  = random.randint(0, negative.shape[1]-1)
            neg = negative[:, index1]
            neg = neg.unsqueeze(1)
            negatives = torch.cat((negatives, neg), dim=1)

            index2 = random.randint(0, positive.shape[1]-1)
            pos = positive[:, index2]
            pos = pos.unsqueeze(1)
            positives = torch.cat((positives, pos), dim=1)

         alpha = mix_ratio * torch.rand(s, device = negative.device)

         negatives = negatives.transpose(1,0)
         positives = positives.transpose(1,0)
         alpha = alpha.unsqueeze(1)
         syn = (alpha * positives)+((1-alpha) * negatives)
         norm = torch.norm(syn, p=2, dim=1, keepdim=True)
         harder_synthetic = syn / norm
         harder_synthetic = harder_synthetic.transpose(1,0)
         return harder_synthetic


def wcreate_set1(negative, s):
    # During warmup, we create synthetic negatives randomly mixing from the queue

         hard_synthetic = torch.tensor([], device=negative.device)
         for i in range(s):
            alpha = torch.rand(1).cuda()
            index1  = random.randint(0, negative.shape[1]-1)
            index2 = random.randint(0, negative.shape[1]-1)
            syn = (alpha * negative[:, index1])+((1-alpha)*negative[:, index2])
            norm = torch.norm(syn, p=2, dim=0, keepdim=True)
            syn = syn / norm
            syn = syn.unsqueeze(1)
            hard_synthetic = torch.cat((hard_synthetic, syn), dim=1)

         return hard_synthetic

def wcreate_set2(positive, negative, s, mix_ratio):
    # During warmup, we create synthetic negatives randomly choosing from the queue
    # and mixing them  with a random positive sample

         harder_synthetic = torch.tensor([], device=negative.device)
         negatives = torch.tensor([], device=negative.device)
         positives = torch.tensor([], device=negative.device)
         positive = positive.transpose(1,0)
         for i in range(s):
            index1  = random.randint(0, negative.shape[1]-1)
            neg = negative[:, index1]
            neg = neg.unsqueeze(1)
            negatives = torch.cat((negatives, neg), dim=1)

            index2 = random.randint(0, positive.shape[1]-1)
            pos = positive[:, index2]
            pos = pos.unsqueeze(1)
            positives = torch.cat((positives, pos), dim=1)

         alpha = mix_ratio * torch.rand(s, device = negative.device)

         negatives = negatives.transpose(1,0)
         positives = positives.transpose(1,0)
         alpha = alpha.unsqueeze(1)
         syn = (alpha * positives)+((1-alpha) * negatives)
         norm = torch.norm(syn, p=2, dim=1, keepdim=True)
         harder_synthetic = syn / norm
         harder_synthetic = harder_synthetic.transpose(1,0)
         return harder_synthetic

