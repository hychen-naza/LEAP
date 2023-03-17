"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from xmlrpc.client import boolean
import random
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
from mingpt.leap_model_babyai import token2idx

AGENT_ID = 10
AGENT_COLOR = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def leap_sample_multi_step(leap_model, env_name, plan_horizon, x, timesteps=None, insts=None, full_obs=None, logger=None, sample_iteration=1, stuck_action=-1):
    leap_model.eval()
    rate = leap_model.rate
    block_size = leap_model.block_size
    batch_size = x.shape[0]
    context = block_size // rate

    cur_timestep = timesteps.cpu().numpy()[0,0,0]
    if (env_name == 'BabyAI-PickupLoc-v0'):
        horizon = random.randint(1, plan_horizon)
    else:
        horizon = plan_horizon
    cur_timestep += horizon
    timesteps=((cur_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to('cuda'))

    init_states = torch.clone(x[:, -1]).unsqueeze(1)

    goals = torch.clone(x[:, -1, -2:]).cpu().to(dtype=torch.float32)
    goals = torch.repeat_interleave(torch.Tensor(goals).unsqueeze(1), horizon - 1, dim=1).to(dtype=torch.long).to('cuda')

    sample_states = [[0,0,0] for _ in range(horizon - 1)]
    sample_states = torch.repeat_interleave(torch.Tensor(sample_states).unsqueeze(0), batch_size, dim=0).to('cuda') 

    if (horizon > 1):
        sample_states = torch.cat((sample_states, goals), dim=2)
        sample_states = torch.cat((init_states, sample_states), dim=1).to(dtype=torch.long)
    elif (horizon == 1):
        sample_states = init_states

    init_obss = torch.clone(full_obs[:,-1]).cpu()
    sample_obss = torch.repeat_interleave(torch.Tensor(init_obss).unsqueeze(1), horizon, dim=1).to(dtype=torch.float32).to('cuda')

    sample_actions = [[token2idx('<-MASK->')] for i in range(horizon)]
    sample_actions = torch.repeat_interleave(torch.Tensor(sample_actions).unsqueeze(0), batch_size, dim=0).to(dtype=torch.long).to('cuda')
    
    # MCMC construct sample trajectory
    for i in range(sample_iteration):
        if (i==0):
            action_masks = np.ones((batch_size, horizon, 1)).astype(boolean)
            action_masks = torch.from_numpy(action_masks).to('cuda')
        else:
            action_masks = np.zeros((batch_size, horizon, 1)).astype(boolean)
            msk = random.randint(0, horizon-1)
            action_masks[0,msk,0] = True
            action_masks = torch.from_numpy(action_masks).to('cuda')

        sample_actions[action_masks] = token2idx('<-MASK->')
        action_logits = leap_model.forward(sample_states, sample_actions, timesteps=timesteps, insts=insts, full_image=sample_obss, mode='eval')
        if (stuck_action != -1):
            action_logits[0,0,stuck_action] = np.float("-inf")
        _, action_val = torch.topk(F.softmax(action_logits.reshape(-1,action_logits.shape[2]), dim=-1), k=1, dim=-1)
        action_val = action_val.reshape(batch_size, horizon)
        if (i < sample_iteration-1):
            sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
        else:
            train_sample_actions = sample_actions.clone()
            train_sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
    leap_model.train()
    return train_sample_actions.flatten().cpu().tolist()

