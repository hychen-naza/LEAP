"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import logging
import random
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

END, PAD, MASK = '<-END->', '<-PAD->', '<-MASK->'
RIGHT = '<-RIGHT->' #[1,0]
LEFT = '<-LEFT->' #[-1,0]
UP = '<-UP->' #[0,-1]
DOWN = '<-DOWN->' #[0,1]
MOVE = '<-MOVE->'
PICKUP = '<-PICKUP->'
DROP = '<-DROP->'
TOGGLE = '<-TOGGLE->'

tokens = [LEFT, RIGHT, MOVE, PICKUP, DROP, TOGGLE, PAD, MASK] #END, 

_token2idx = dict(zip(tokens, range(len(tokens))))
def token2idx(token):
    return _token2idx[token]

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.ones(config.block_size + 1, config.block_size + 1)
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, is_debug=False):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, is_debug):
        x = x + self.attn(self.ln1(x), is_debug=is_debug)
        x = x + self.mlp(self.ln2(x))
        return x


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class SimpleResidualBlock(nn.Module):
    def __init__(self, input_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        if stride == 1:
            if (input_channel_size == out_channel_size):
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride),
                                        nn.Conv2d(out_channel_size, out_channel_size, kernel_size=1, stride=stride))
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu2(out + shortcut)        
        return out


class LEAP_GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_type = config.model_type
        self.sample_iteration = config.sample_iteration
        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = [] #nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        for i in range(config.n_layer):
            block = Block(config)
            self.blocks.append(block) 
            self.add_module('block_' + str(i), block)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.action_head = nn.Linear(config.n_embd, config.vocab_size-2, bias=False) #don't consider the PAD and MASK
        self.block_size = config.block_size
        self.env_size = config.env_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if (self.env_size in [16,19]):
            self.state_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                SimpleResidualBlock(128, 128, 2)
            )
        elif (self.env_size in [7,8,9,10,13]):
            self.state_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        else:
            pdb.set_trace()

        self.state_embeddings = nn.Sequential(nn.Embedding(config.env_size, config.n_embd))
        self.state_embeddings_linear = nn.Sequential(nn.Linear(config.n_embd*config.state_dim, config.n_embd*2), 
                                                    nn.Tanh(),
                                                    nn.Linear(config.n_embd*2, config.n_embd*2),
                                                    nn.Tanh(),
                                                    nn.Linear(config.n_embd*2, config.n_embd))
        self.direction_embeddings = nn.Sequential(nn.Embedding(4, config.n_embd))

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.word_embedding = nn.Embedding(100, config.n_embd)
        self.instr_rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)

        self.controllers = []
        num_module = 2
        for ni in range(num_module):
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd, out_features=config.n_embd,
                    in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni), mod)
        self.agent_pos_controllers = [] # fuse the fully observable image with agent goal position and current position
        
        for ni in range(num_module): #
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd, out_features=config.n_embd,
                    in_channels=128, imm_channels=128)
            self.agent_pos_controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni+num_module), mod)

        if (self.env_size in [13,19]):
            self.film_pool = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1152, config.n_embd), 
                nn.ReLU())
        elif (self.env_size in [7,8,9,10,16]):
            self.film_pool = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, config.n_embd), 
                nn.ReLU())
        else:
            pdb.set_trace()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d, torch.nn.MaxPool2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif (pn.find('weight') != -1 and isinstance(m, whitelist_weight_modules)): #and isinstance(m, whitelist_weight_modules)
                    decay.add(fpn)
                elif (pn.find('bias') != -1 and fpn != 'FiLM_Controler_0.bias.weight' and fpn != 'FiLM_Controler_1.bias.weight' and 
                        fpn != 'FiLM_Controler_2.bias.weight' and fpn != 'FiLM_Controler_3.bias.weight'): # and isinstance(m, blacklist_weight_modules)
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('instr_rnn.bias_hh_l0')
        no_decay.add('instr_rnn.bias_ih_l0')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _get_instr_embedding(self, instr):
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]

    def mcmc_construct_trajectory(self, init_states, init_obss, target_states, target_imgs, t, insts, \
            masked_actions = None, target_actions=None, target_action_masks=None, task_name=None):
        sample_iteration = 1 #self.sample_iteration
        self.rate = 2
        self.batch_size = init_states.shape[0]
        context = self.block_size // self.rate
        
        goals = init_states[:, -2:].to(dtype=torch.float32)
        goals = torch.repeat_interleave(torch.Tensor(goals).unsqueeze(1), context - 1, dim=1).to(dtype=torch.long).to('cuda')

        sample_states = [[0,0,0] for _ in range(context - 1)]
        sample_states = torch.repeat_interleave(torch.Tensor(sample_states).unsqueeze(0), self.batch_size, dim=0).to('cuda') #.to(dtype=torch.float32)
        if (context > 1):
            sample_states = torch.cat((sample_states, goals), dim=2)
            init_states = init_states.unsqueeze(1).to('cuda')
            sample_states = torch.cat((init_states, sample_states), dim=1).to(dtype=torch.long)
        elif (context == 1):
            sample_states = init_states.unsqueeze(1).to('cuda')
        
        sample_obss = torch.repeat_interleave(torch.Tensor(init_obss).unsqueeze(1), context, dim=1).to(dtype=torch.float32)
        sample_obss = sample_obss.to('cuda')

        sample_actions = [[token2idx('<-MASK->')] for i in range(context)]
        sample_actions = torch.repeat_interleave(torch.Tensor(sample_actions).unsqueeze(0), self.batch_size, dim=0).to(dtype=torch.long).to('cuda')
        sample_actions = torch.clone(masked_actions)
        init_temperature = 0.5
        energys = []
        # MCMC construct sample trajectory
        for i in range(sample_iteration):
            temperature = min(init_temperature + i / (2*sample_iteration), 0.9) if i > 0 else 0
            action_masks = np.random.uniform(0, 1, (self.batch_size, context, 1)) >= temperature
            action_masks = torch.from_numpy(action_masks).to('cuda')

            action_logits = self.forward(sample_states, sample_actions, timesteps=t, insts=insts, full_image=sample_obss, task_name=task_name)
            _, action_val = torch.topk(F.softmax(action_logits.reshape(-1,action_logits.shape[2]), dim=-1), k=1, dim=-1)
            action_val = action_val.reshape(self.batch_size, context)
            if (i < sample_iteration-1):
                sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
                iter_actions = sample_actions.cpu().numpy()
                sample_states, sample_obss = self.update_samples(sample_states, sample_obss, iter_actions)
                eng, _, _ = self.compute_energy(target_actions, sample_actions, action_logits, torch.logical_and(action_masks, target_action_masks))
                energys.append(eng.item())
            else:
                train_sample_actions = sample_actions.clone()
                train_sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
                eng, _, _ = self.compute_energy(target_actions, train_sample_actions, action_logits, torch.logical_and(action_masks, target_action_masks))
                energys.append(eng.item()) 

        targets = target_actions.masked_select(target_action_masks)
        samples = train_sample_actions.masked_select(target_action_masks)
        action_correct_rate = torch.sum(targets == samples) / len(samples)
        action_correct_rate_steps = []
        for i in range(context):
            step_mask = np.zeros(target_action_masks.shape)
            step_mask[:,i] = 1
            step_mask = torch.from_numpy(step_mask).to(torch.bool).to('cuda')
            targets = target_actions.masked_select(torch.logical_and(target_action_masks,step_mask))
            samples = train_sample_actions.masked_select(torch.logical_and(target_action_masks,step_mask))
            if (len(targets)>0):
                action_correct_rate_steps.append((torch.sum(targets == samples) / len(samples)).item())
            else:
                action_correct_rate_steps.append(0)
        return train_sample_actions, action_logits, energys[-1], action_correct_rate.item(), action_correct_rate_steps

    def compute_energy(self, target_actions, sample_actions, action_logits, action_masks, use_sum=False, use_neg_energy=False):
        action_size = action_logits.shape[2]
        action_logits = action_logits.masked_select(action_masks).reshape(-1, action_size)
        log_softmax_action_logits = F.log_softmax(action_logits, dim=-1)
        pos_action_targets = target_actions.masked_select(action_masks).reshape(-1)
        pos_action_energy = -torch.gather(log_softmax_action_logits, 1, pos_action_targets.unsqueeze_(dim=1))
        if (use_sum):
            pos_energy = torch.sum(pos_action_energy) 
        else:
            pos_energy = torch.sum(pos_action_energy) 
        energy = torch.clone(pos_energy) 
        neg_energy = torch.Tensor([0])
        
        if (use_sum):
            return energy, len(action_logits)
        else:
            return energy, pos_energy, -neg_energy
     
    def train_step(self, target_states, target_actions, target_imgs, state_masks=None, \
            timesteps=None, insts=None, init_states=None, init_obss=None,\
            mode='train', rtgs=None, logger=None, task_name=None):
        plan_horizon = target_actions.shape[1]
        total_loss = 0
        for i in range(plan_horizon):
            action_masks = target_actions[:,i:i+1] != token2idx('<-PAD->')
            action_masks = torch.repeat_interleave(action_masks, plan_horizon, dim=1)
            action_masks[:,:i] = False
            action_masks[:,i+1:] = False
            msked_actions = torch.clone(target_actions)
            msked_actions[action_masks] = token2idx('<-MASK->')
            sample_actions, train_action_logits, mcmc_energy, correct_action_rate, action_correct_rate_steps = \
                self.mcmc_construct_trajectory(init_states, init_obss, target_states, target_imgs, timesteps, insts, \
                    masked_actions = msked_actions, target_actions=target_actions, target_action_masks=action_masks, task_name=task_name)
            
            energy, pos_energy, neg_energy = self.compute_energy(target_actions, sample_actions, train_action_logits, action_masks, use_neg_energy=False)
            gt_traj_energy = 0 #self.gt_trajectory_energy(target_states, target_imgs, timesteps, insts, target_actions=target_actions, target_action_masks=action_masks, task_name=task_name)
            loss = energy
            total_loss += loss
        return total_loss, gt_traj_energy, mcmc_energy, correct_action_rate, action_correct_rate_steps, pos_energy, neg_energy


    def gt_trajectory_energy(self, target_states, target_imgs, t, insts, target_actions=None, target_action_masks=None, sample_iteration=5, task_name=None):
        self.rate = 2
        self.batch_size = target_states.shape[0]
        context = self.block_size // self.rate
        # computer gt energy during iteration
        length = target_states.shape[1]
        gt_traj_energy = 0
        nums = 0
        for i in range(length):
            action_masks = np.zeros((self.batch_size, length, 1))
            action_masks[:,i] = 1
            action_masks = torch.from_numpy(action_masks).to(dtype=torch.bool).to('cuda')
            sample_actions = torch.clone(target_actions)
            sample_actions[action_masks] = token2idx('<-MASK->')
            action_logits = self.forward(target_states, sample_actions, timesteps=t, insts=insts, full_image=target_imgs, task_name=task_name)
            _, action_val = torch.topk(F.softmax(action_logits.reshape(-1,action_logits.shape[2]), dim=-1), k=1, dim=-1)
            action_val = action_val.reshape(self.batch_size,length)
            sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
            energy, energy_count = self.compute_energy(target_actions, sample_actions, action_logits, torch.logical_and(action_masks, target_action_masks), use_sum=True)
            gt_traj_energy += energy.item()
            nums += energy_count
        gt_traj_energy /= nums
        return gt_traj_energy

    def forward(self, states, actions, rtgs=None, timesteps=None, insts=None, full_image=None, mode='train', is_debug=False, task_name=None):
        
        image_embeddings = self.state_encoder(full_image.reshape(-1, 3, self.env_size, self.env_size).type(torch.float32).contiguous()) # (batch * block_size, n_embd)

        instr_embedding = self._get_instr_embedding(insts)
        instr_embedding = torch.repeat_interleave(instr_embedding.unsqueeze(1), states.shape[1], dim=1)
        instr_embedding = instr_embedding.reshape(-1, instr_embedding.size(2))

        state_embeddings = self.state_embeddings(states[:,:,:2]) 
        goal_embeddings = self.state_embeddings(states[:,:,-2:]) 
        direction_embeddings = self.direction_embeddings(states[:,:,2:3]) 
        state_embeddings = torch.cat((state_embeddings, direction_embeddings, goal_embeddings), dim=2).reshape(states.shape[0]*states.shape[1], -1)
        state_embeddings = self.state_embeddings_linear(state_embeddings)
 
        # jointly learn the image and instr
        for controler in self.controllers:
            image_embeddings = controler(image_embeddings, instr_embedding)

        for controller in self.agent_pos_controllers:
            image_embeddings = controller(image_embeddings, state_embeddings)

        state_embeddings = self.film_pool(image_embeddings)        
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)

        actions = None if task_name == 'BabyAI-PickupLoc-v0' else actions        
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1]:,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1]:,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x, is_debug)
        x = self.ln_f(x)
        action_logits = self.action_head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            action_logits = action_logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            action_logits = action_logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            action_logits = action_logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            action_logits = action_logits # for completeness
        else:
            raise NotImplementedError()
        return action_logits 
