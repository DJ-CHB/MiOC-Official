# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.svm import OneClassSVM
from utils_mioc import create_syn_neg, wcreate_set1, wcreate_set2
import random
import numpy as np

class MoCo(LightningModule):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.trainer.strategy.all_gather(keys)
        if keys.dim() > 2:
            keys = keys.flatten(end_dim=1)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"{self.K=} {batch_size=}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.trainer.strategy.all_gather(x)
        if x_gather.dim() > 4:
            x_gather = x_gather.flatten(end_dim=1)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        self.trainer.strategy.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = self.trainer.global_rank
        idx_this = idx_shuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.trainer.strategy.all_gather(x)
        if x_gather.dim() > 2:
            x_gather = x_gather.flatten(end_dim=1)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = self.trainer.global_rank
        idx_this = idx_unshuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_mochi: bool, step: int):
        if not is_mochi:
            return self.forward_moco(im_q, im_k)
        else:
            is_warmup = self.current_epoch < self.args.mochi_warmup
            return self.forward_mioc(im_q, im_k, is_warmup=is_warmup, step=step)

    def forward_moco(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # [b dim] [b dim]
        # negative logits: NxK
        l_neg = torch.einsum( "nc,ck->nk", [q, self.queue.clone().detach()])  # [b dim] [dim K]

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_mioc(self, im_q, im_k, is_warmup: bool, step: int):
        """
        MiOC Forward pass
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        clf = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.01)
        positive = torch.cat((q, k), dim=0)
        positive = positive.detach().cpu().numpy()
        clf.fit(positive)
        n = self.queue.detach().cpu().numpy().T
        neg_points = clf.predict(n)
        neg_points_inside = self.queue[:, neg_points==1]

        # region 4.1 Mixing the hardest negatives
        if is_warmup:
            S = wcreate_set1(negative=self.queue, s=self.args.mochi_s)
            S_prime = wcreate_set2(positive=q, negative=self.queue, s=self.args.mochi_s_prime, mix_ratio=0.5)
        else:
            S = create_syn_neg(positive=q, negative=self.queue, s=self.args.mochi_s,mix_ratio=0.5 ) # generate harder negatives from the queue
            S_prime = create_syn_neg(positive=q, negative=neg_points_inside, s=self.args.mochi_s_prime, mix_ratio=0.5) # generate harder negatives from the inner OCSVM samples




        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        # negative are in H and H_prime
        negatives = torch.cat((self.queue, S, S_prime), dim=1).detach()
        l_neg = torch.einsum("nc,ck->nk", q, negatives)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)


        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
