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

from LLM import GPT2 as actor 
from LLM import GPT2 as critic  
from reward import model as RewardModelGPT2
from ActorCritic import ActorCritic

from utils import masked_mean, gumbel_sample, get_optimizer

# data structure used in training

Memory = namedtuple('Memory', [
    'sequence',
    'prompt_mask',
    'mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])

Memory = namedtuple('Memory', [
    'sequence',
    'prompt_mask',
    'mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])

@beartype
class ExperienceDataset(Dataset):
    def __init__(
        self,
        data: List[torch.Tensor],
        device = None
    ):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))

def create_dataloader(data, batch_size, shuffle = True, device = None, **kwargs):
    ds = ExperienceDataset(data, device = device)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle, **kwargs)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def masked_normalize(t, eps = 1e-5, mask = None, dim = None):
    dim = default(dim, tuple(range(t.ndim)))
    kwargs = dict(dim = dim, keepdim = True)

    mean = masked_mean(t, mask = mask, **kwargs)
    mean_centered = t - mean
    var = masked_mean(mean_centered ** 2, mask = mask, **kwargs)

    return mean_centered * var.clamp(min = eps).rsqrt()

def pad_sequence_fixed(sequences, *args, **kwargs):
    first_el = sequences[0]
    has_no_dimension = first_el.ndim == 0

    # if no dimensions, add a single dimension
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))

    out = pad_sequence(sequences, *args, **kwargs)

    if has_no_dimension:
        out = rearrange(out, '... 1 -> ...')

    return out

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def log_prob(prob, indices):
    assert prob.shape[:2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)

def shift(t, value = 0, shift = 1, dim = -1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value = value)

def masked_entropy(prob, dim = -1, mask = None):
    entropies = (prob * log(prob)).sum(dim = -1)
    return masked_mean(entropies, mask = mask).mean()

def masked_kl_div(prob1, prob2, mask = None, reduce_batch = False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim = -1)
    loss = masked_mean(kl_divs, mask)

    if reduce_batch:
        return loss.mean()

    return loss

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


@beartype
class RLHF(nn.Module):
    def __init__(
        self,
        *,
        prompts: Optional[List[str]] = None,
        prompts_path: Optional[str] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
        tokenizer: Callable = None,
        reward_model: RewardModelGPT2,  #TO DO check for signatures
        critic = Optional[GPT2],
        actor_critic: Optional[ActorCritic] = None,
        actor_lr = 1e-4,                            #TODO: refactor to load from config
        critic_lr = 1e-4,
        actor_wd = 0.,
        critic_wd = 0.,
        actor_adam_eps = 1e-7,
        critic_adam_eps = 1e-7,
        actor_lora = False,
        critic_lora = False,
        actor_lora_r = 8,
        critic_lora_r = 8,
        critic_pooled_values = True,
        actor_dropout = 0.,
        critic_dropout = 0.,
        betas = (0.9, 0.999),
        max_norm = None,
        eps_clip = 0.2,
        value_clip = 0.4,
        beta_s = .01,
        pad_value = 0.,
        minibatch_size = 16,
        epochs = 1,
        kl_div_loss_weight = 0.1, # between old action probs and new action probs - not sure what the right value is
        accelerate_kwargs: dict = {},
        use_lion = False
    ):
        super().__init__()

        # take care of prompts -> token ids

        assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        if exists(prompts_path):
            path = Path(prompts_path)
            prompts = path.read_text().split('\n')

        if exists(prompts):
            assert len(prompts) > 0, 'no prompts'
            assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'
            prompt_token_ids = tokenizer(prompts)

        self.pad_value = pad_value # token pad value
        self.num_prompts = prompt_token_ids.shape[0]
        self.register_buffer('prompt_token_ids', prompt_token_ids)

        if not exists(actor_critic):
            actor_critic = ActorCritic(
                actor = actor,
                critic = critic,
            ).to(config.device)

        self.actor_critic = actor_critic

        self.reward_model = reward_model.eval()

        # train hyperparameters
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_norm = max_norm

        self.kl_div_loss_weight = kl_div_loss_weight

        # optimizers

        self.actor_optim = get_optimizer(actor_critic.actor_parameters(), lr = actor_lr, wd = actor_wd, betas = betas, eps = actor_adam_eps, use_lion = use_lion)
        self.critic_optim = get_optimizer(actor_critic.critic_parameters(), lr = critic_lr, wd = critic_wd, betas = betas, eps = critic_adam_eps, use_lion = use_lion)

        # ppo hyperparams

        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.beta_s = beta_s


    def print(self, msg):
        return self.accelerate.print(msg)

    def save(self, filepath = './checkpoint.pt'):     
        torch.save(self.actor_critic.state_dict(), filepath)

    def load(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.actor_critic.load_state_dict(state_dict)

    @property
    def device(self):
        return self.accelerate.device

    @torch.no_grad()
    def generate(
        self,
        max_seq_len,
        *args,
        prompt,
        num_samples = 4,  # sample 4 per prompt and select the one with highest reward
        **kwargs
    ):
        assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
        prompt = repeat(prompt, 'n -> b n', b = num_samples)

        actor_critic = self.actor_critic
        reward_model = self.reward_model

        actor_critic.eval()

        (
            actions,
            sequences,
            mask,
            prompt_mask,
            action_logits,
            _
        ) = actor_critic.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            return_values = False,
            **kwargs
        )

        rewards = reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask,
            sample = True
        )

        best_sequence_index = rewards.topk(1, dim = -1).indices  #to do work on the new reward model

        best_sequence = sequences[best_sequence_index]
        best_sequence = rearrange(best_sequence, '1 ... -> ...')

        return best_sequence
    

    def learn(
        self,
        memories: Deque[Memory]
    ):
        # stack all data stored in the memories

        all_memories_stacked_and_padded = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))

        # prepare dataloader for policy phase training

        dl = create_dataloader(all_memories_stacked_and_padded, self.minibatch_size, device = self.device)

        self.actor_critic.train()

        # PPO training

        for _ in range(self.epochs):
            for (
                sequences,
                prompt_masks,
                masks,
                old_action_probs,
                old_log_probs,
                rewards,
                old_values
            ) in dl:
                action_masks = ~prompt_masks & masks

                action_logits, values = self.actor_critic(
                    sequences,
                    mask = action_masks
                )

                action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
                action_len = old_log_probs.shape[-1]

                action_probs = action_logits.softmax(dim = -1)
                action_log_probs = log_prob(action_probs, sequences)
                action_log_probs = action_log_probs[:, -action_len:]

                # calculate entropies, taking into account which part of the sequence is actually an action

                entropies = masked_entropy(action_probs, mask = action_masks)

                # calculate kl div between old action probs and new ones, taking into account which part of the sequence is action or not

                kl_penalty = 0.

                if self.kl_div_loss_weight > 0:
                    kl_penalty = masked_kl_div(old_action_probs, action_probs, mask = action_masks) * self.kl_div_loss_weight

                # subtract the kl penalty from the rewards

                rewards = rewards - kl_penalty

                # handle non-pooled values

                normalize_kwargs = dict()

                if old_values.ndim == 2:
                    old_values, values = map(lambda t: shift(t, shift = 1, dim = -2), (old_values, values))

                    old_values = old_values[:, -action_len:]
                    values = values[:, -action_len:]
                    rewards = rearrange(rewards, 'b -> b 1')
                    normalize_kwargs = dict(dim = -1, mask = action_masks[:, -action_len:])

                if values.ndim < rewards.ndim:
                    values = rearrange(values, '... -> ... 1')

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = masked_normalize(rewards - old_values, **normalize_kwargs)

                if advantages.ndim == 1:
                    advantages = rearrange(advantages, 'b -> b 1')

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropies

                # combine losses
                loss = policy_loss.mean()

                # update actor with loss
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

                torch.cuda.synchronize(device)

                # compute value loss
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -critic_eps_clip, critic_eps_clip
                )
                value_loss1 = (value_loss_clipped - rewards) ** 2
                value_loss2 = (values - rewards) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError("Value loss is nan")
                print("value_loss", value_loss.item())

                # upate critic with loss
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step() 
                self.actorcritic.eval()
            



    def train(
        self,
        num_episodes = 50000,
        max_timesteps = 500,
        update_timesteps = 5000,
        max_batch_size = 16,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.
    ):
        device = self.device

        time = 0
        memories = deque([])

        for eps in tqdm(range(num_episodes), desc = 'episodes'):
            for timestep in range(max_timesteps):
                time += 1

                # select a bunch of random states (prompts)
                # and get the action (sampled sequence from palm as well as the action probs)
                # also calculate the reward using reward model and store

                rand_prompt_index = randrange(0, self.num_prompts)

                state = self.prompt_token_ids[rand_prompt_index]

                # remove padding from state

                state_mask = state != self.pad_value
                state = state[state_mask]

                # get predicted sequence

                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    action_logits,
                    value
                ) = self.actor_critic.generate(          #TODO CHANGE THIS
                    rearrange(state, 'n -> 1 n'),
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                    return_values = True
                )
                action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token

                action_prob = action_logits.softmax(dim = -1)

                action_len = actions.shape[-1]
                action_log_prob = log_prob(action_prob, sequence)
                action_log_prob = action_log_prob[:, -action_len:]

                actions = rearrange(actions, '1 ... -> ...')

                # get reward as given by supervised trained reward model

                sequence = torch.cat((state, actions), dim = 0)

                prompt_length = len(state)
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length

                sequence = rearrange(sequence, 'n -> 1 n')
                prompt_mask = rearrange(prompt_mask, 'n -> 1 n')
                mask = default(mask, lambda: torch.ones(sequence.shape, dtype = torch.bool, device = device))

                reward = self.reward_model(
                    sequence,
                    prompt_mask = prompt_mask,
                    mask = mask,
                    sample = True
                )

                detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')

                # store memory for learning

                memories.append(Memory(*map(detach_to_cpu_, (
                    sequence,
                    prompt_mask,
                    mask,
                    action_prob,
                    action_log_prob,
                    reward,
                    value
                ))))

                # learn from the stored memories

                if time % update_timesteps == 0:
                    self.learn(memories)
                    memories.clear()

        print('rlhf training complete')




