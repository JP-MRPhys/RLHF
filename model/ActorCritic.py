import math
from pathlib import Path
import copy
from tqdm import tqdm
from functools import partial
from collections import deque, namedtuple
from random import randrange

from beartype import beartype
from beartype.typing import List, Optional, Callable, Deque

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from LLM import GPT2
from reward import model as RewardModelGPT2

from utils import masked_mean, gumbel_sample, get_optimizer

#add config



# actor critic

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])




#TODO: ADD CONFIG and change the LORA and other parameter for tiding this up.. understand how to fine tune GPT2
@beartype
class ActorCritic(nn.Module):
    def __init__(
        self,
        pooled_values=False          #TODO: check this

    ):
        super().__init__()
        self.actor = GPT2()  # add GPT2 config    #actor is LLM
        self.critic = GPT2() #critic is LLM
        self.dim=GPT2.dim


        self.pooled_values = pooled_values
        self.value_head = nn.Sequential(
            nn.Linear(self.LLM.dim, 1),
            Rearrange('... 1 -> ...')
        )

        nn.init.zeros_(self.value_head[0].bias)
        nn.init.orthogonal_(self.value_head[0].weight, gain = math.sqrt(2))

    def actor_parameters(self):
            return self.actor.parameters()


    def critic_parameters(self):
            return [*self.critic.parameters(), *self.value_head.parameters()]

    #state are the prompts and we generate actions over the state via passing througth actor model
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state,
        max_seq_len=None,
        eos_token = None,
        return_values = False,
        **kwargs
    ):
        
        #TODO Add maks        
        actions=self.actor.generate(state,state_mask=None)

        #TODO match the input outputs

        """
        actions = self.actor.generate(  
            max_seq_len,
            prompt = state,       
            eos_token = eos_token,     
            finetune_scope = self.actor_lora_scope,
            use_tqdm = True,
            **kwargs
        )
        """

        sequence = torch.cat((state, actions), dim = -1)
        action_len = actions.shape[-1]
        state_len = state.shape[-1]

        prompt_mask = torch.arange(sequence.shape[-1], device = state.device) < state_len
        prompt_mask = repeat(prompt_mask, 'n -> b n', b = sequence.shape[0])

        action_mask = ~prompt_mask

        mask = None
        if exists(eos_token):
            mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
            mask = F.pad(mask, (1, -1), value = True) # include eos token
            action_mask &= mask

        action_logits, value = self.forward(
            sequence,
            state_mask = action_mask,
            return_values = return_values   
        )        

        return PPOActionCriticReturn(
            actions,
            sequence,
            mask,
            prompt_mask,
            action_logits,
            value
        )

    def forward(
        self,
        x,
        mask = None,
        return_values = True
    ):
        action_logits = self.actor(
            x, mask)

        if not return_values:
            return action_logits, None

        critic_embeds = self.critic(
            x,
            return_only_embedding = True,                #TO DO check this in critic fucntion
        )

        if self.pooled_values:
            critic_embeds = shift(critic_embeds, shift = 1, dim = -2)
            critic_embeds = masked_mean(critic_embeds, mask, dim = 1)  #TODO check this

        values = self.value_head(critic_embeds)

        return action_logits, values



# data

