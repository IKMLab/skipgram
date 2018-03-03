"""Fully implemented classes from the task."""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import dataset, dataloader
import pandas as pd
import nltk
import os
import glovar
import collections


def get_data_loader(data, batch_size, collate_fn):
    return dataloader.DataLoader(data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=collate_fn)


class NegativeSampler:

    def __init__(self, vocab, frequencies, contexts, num_negs):
        """Create a new NegativeSampler.

        Args:
          vocab: Dictionary.
          frequencies: List of integers, the frequencies of each word,
            sorted in word index order.
          contexts: Dictionary.
          num_negs: Integer, how many to negatives to sample.
        """
        self.vocab = vocab
        self.n = len(vocab)
        self.contexts = contexts
        self.num_negs = num_negs
        self.distribution = self.p(list(frequencies.values()))

    def __call__(self, tok_ix):
        """Get negative samples.

        Args:
          tok_ix: Integer, the index of the center word.

        Returns:
          List of integers.
        """
        neg_samples = np.random.choice(
            self.n,
            size=self.num_negs,
            p=self.distribution)
        # make sure we haven't sampled center word or its context
        invalid = [-1, tok_ix] + list(self.contexts[tok_ix])
        for i, ix in enumerate(neg_samples):
            if ix in invalid:
                new_ix = -1
                while new_ix in invalid:
                    new_ix = np.random.choice(self.n,
                                              size=1,
                                              p=self.distribution)[0]
                neg_samples[i] = new_ix
        return [int(s) for s in neg_samples]

    def p(self, freqs):
        """Determine the probability distribution for negative sampling.

        Args:
          freqs: List of integers.

        Returns:
          numpy.ndarray.
        """
        ### Impelement Me ###
        freqs = np.array(freqs)
        return np.power(freqs, 3 / 4) / np.sum(np.power(freqs, 3 / 4))


class Collate:

    def __init__(self, neg_sampler):
        self.sampler = neg_sampler

    def __call__(self, pairs):
        ### Implement Me ###
        batch_size = len(pairs)
        centers = [x[0] for x in pairs]
        contexts = [x[1] for x in pairs]
        context_and_negs = []
        for i in range(batch_size):
            neg_samples = self.sampler(centers[i])
            context_and_negs.append([contexts[i]] + list(neg_samples))
        return centers, context_and_negs


class SkipGram(nn.Module):
    """SkipGram Model."""

    def __init__(self, vocab, emb_dim, num_negs, lr):
        """Create a new SkipGram.

        Args:
          vocab: Dictionary, our vocab dict with token keys and index values.
          emb_dim: Integer, the size of word embeddings.
          num_negs: Integer, the number of non-context words to sample.
          lr: Float, the learning rate for gradient descent.
        """
        super(SkipGram, self).__init__()
        self.vocab = vocab
        self.n = len(vocab)  # size of the vocab
        self.emb_dim = emb_dim
        self.num_negs = num_negs

        ### Implement Me: define V and U ###

        ### Implement Me: initialize V and U with unform distribution in [-0.01, 0.01] ###

        self.V = nn.Embedding(self.n, emb_dim)
        self.U = nn.Embedding(self.n, emb_dim)
        # self.V = nn.Parameter(torch.Tensor(self.n, emb_dim), requires_grad=True)
        # self.U = nn.Parameter(torch.Tensor(emb_dim, self.n), requires_grad=True)
        nn.init.uniform(self.V.weight, a=-0.01, b=0.01)
        nn.init.uniform(self.U.weight, a=-0.01, b=0.01)

        # Adam is a good optimizer and will converge faster than SGD
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch):
        """Compute the forward pass of the network.

        1. Lookup embeddings for center and context words
            - use self.lookup()
        2. Sample negative samples
            - use self.negative_samples()
        2. Calculate the probability estimates
            - make sure to implement and use self.softmax()
        3. Calculate the loss
            - using self.loss()

        Args:
          batch: whatever data structure you decided to make for a batch.

        Returns:
          loss (torch.autograd.Variable).
        """
        ### Implement Me: lookup center words ###
        ### Implement Me: lookup context words ###
        ### Implement Me: lookup negative words ###
        cents, conts_negs = batch
        cents = self.V(self.lookup_tensor(cents))
        conts_negs = self.U(self.lookup_tensor(conts_negs)).permute([0, 2, 1])
        logits = cents.matmul(conts_negs)
        preds = self.softmax(logits)
        targets = self.targets(cents.shape[0])
        return self.loss(preds, targets)

    def lookup_tensor(self, indices):
        """Lookup embeddings given indices.

        Args:
          embedding: nn.Parameter, an embedding matrix.
          indices: List of integers, the indices to lookup.

        Returns:
          torch.autograd.Variable of shape [len(indices), emb_dim]. A matrix
            with horizontally stacked word vectors.
        """
        if torch.cuda.is_available():
            return Variable(torch.LongTensor(indices),
                            requires_grad=False).cuda()
        else:
            return Variable(torch.LongTensor(indices),
                            requires_grad=False)

    def loss(self, preds, targets):
        """Compute cross-entropy loss.

        Implement this for practice, don't use the built-in PyTorch function.

        Args:
          preds: Tensor of shape [batch_size, vocab_size], our predictions.
          targets: List of integers, the vocab indices of target context words.
        """
        ### Implement Me ###
        return -1 * torch.sum(targets * torch.log(preds))

    def optimize(self, loss):
        """Optimization step.

        Args:
          loss: Scalar.
        """
        # Remove any previous gradient from our tensors before calculating again.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def softmax(self, logits):
        """Compute the softmax function.

        Implement this for practice, don't use the built-in PyTorch function.

        Args:
          logits: Tensor of shape [batch_size, vocab_size].

        Returns:
          Tensor of shape [batch_size, vocab_size], our predictions.
        """
        ### Impelement Me ###
        return torch.exp(logits) / torch.sum(torch.exp(logits))

    def targets(self, batch_size):
        """Get the conventional targets for the batch.

        Args:
          batch_size: Integer.

        Returns:
          torch.LongTensor.
        """
        targets = torch.zeros((batch_size, self.num_negs + 1))
        targets[:, 0] = 1
        return Variable(targets, requires_grad=False)

