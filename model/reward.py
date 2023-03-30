from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
import tensorflow as tf
import config
from einops.layers.torch import Rearrange

#add config 
#device
#model path 



import copy
from pathlib import Path

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from LLM import GPT2

from utils import masked_mean, gumbel_sample


# helper functions


def exists(val):
    return val is not None

# Reward Model - LLM with a scalar head

@beartype
class RewardModelGPT(nn.Module):
    def __init__(
        self,
        reward_lora_scope = 'reward',
        num_binned_output=0.,
    ):
        super().__init__()

        self.GPT2=GPT2()

        dim = GPT2.dim

        self.binned_output = num_binned_output > 1
        #(batch size, word, embedding_dim)
        self.prompt_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.response_embed = nn.Parameter(torch.zeros(1, 1, dim))

        if self.binned_output:
            self.to_pred = nn.Linear(dim, num_binned_output)
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, 1, bias = False),
                Rearrange('... 1 -> ...')
            )

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def finetune_parameters(self):
        return [
            *self.to_pred.parameters(),
            *(self.GPT2.parameters())
        ]

    def forward(
        self,
        x,
        mask = None,
        prompt_mask = None,
        prompt_lengths = None,
        labels = None,
        sample = False,
        sample_temperature = 1.,
        disable_lora = True
    ):

        assert not (exists(prompt_mask) and exists(prompt_lengths))

        # derive prompt mask from prompt lengths

        if exists(prompt_lengths):
            batch, seq_len = x.shape
            arange = torch.arange(seq_len, device = x.device)
            prompt_mask = repeat(arange, 'n -> b n', b = batch) < rearrange(prompt_lengths, 'b -> b 1')

        # reward model should have an understanding of which section is prompt, and which section is response

        extra_embed = None

        if exists(prompt_mask):
            extra_embed = torch.where(
                rearrange(prompt_mask, 'b n -> b n 1'),
                self.prompt_embed,
                self.response_embed
            )

        # get embeddings from LLM
        embeds=self.GPT2.embedding(x) # TODO add condition to check dimensions   

        ###
        pooled = masked_mean(embeds, mask, dim = 1)
        pred = self.to_pred(pooled)

        if sample and self.binned_output:
            assert not exists(labels)
            pred = gumbel_sample(pred, temperature = sample_temperature, dim = -1)

        if not exists(labels):
            return pred

        if not self.binned_output:
            return F.mse_loss(pred, labels)

        return F.cross_entropy(pred, labels)
