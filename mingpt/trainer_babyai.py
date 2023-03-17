"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import math
import gym
import logging
from tqdm import tqdm
import numpy as np
import pdb
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import leap_sample_multi_step, AGENT_ID, AGENT_COLOR
from collections import deque
import random
import cv2
import torch
from PIL import Image
import logging
from babyai.utils.agent import BotAgent

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.ckpt_path = './model.pkl'

class Trainer:

    def __init__(self, leap_model, leap_train_dataset, test_dataset, config, env, env_name, rate, plan_horizon, sample_iteration, inst_preprocessor, env_size):
        self.leap_model = leap_model
        self.leap_train_dataset = leap_train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.env = env
        self.env_name = env_name
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.sample_iteration = sample_iteration
        self.inst_preprocessor = inst_preprocessor
        self.env_size = env_size
        self.bot_advisor_agent = BotAgent(self.env)
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.leap_model = self.leap_model.to(self.device)
        console = logging.StreamHandler(sys.stdout)
        console_log_level = 100
        console.setLevel(console_log_level)
        self.logger = logging.getLogger(__name__)  
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'log_file_{self.plan_horizon}.log')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def save_checkpoint(self):
        print(f"saving {self.config.ckpt_path}")
        torch.save(self.leap_model, self.config.ckpt_path)    
        
    def load_checkpoint(self):
        print(f"loading {self.config.ckpt_path}")
        self.leap_model = torch.load(self.config.ckpt_path)
        self.leap_model.eval()

    def train(self):
        leap_model, config = self.leap_model, self.config

        raw_leap_model = leap_model.module if hasattr(self.leap_model, "module") else leap_model
        leap_optimizer = raw_leap_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            leap_model.train(is_train)
            leap_data = self.leap_train_dataset if is_train else self.test_dataset
            leap_loader = DataLoader(leap_data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=0) #config.num_workers
            leap_losses = []
            gt_traj_energys = []
            mcmc_energys = []
            free_rates = []
            action_correct_rates = []
            all_action_correct_rate_steps = []
            pos_energies, neg_energies = [], []

            #pbar = tqdm(enumerate(leap_loader), total=len(leap_loader)) if is_train else enumerate(leap_loader)
            for it, (x, y, m_y, full_imgs, msk_x, msk_y, r, t, inst, init_x, init_image) in enumerate(leap_loader):

                x = x.to(self.device)
                y = y.to(self.device)
                m_y = m_y.to(self.device)
                full_imgs = full_imgs.to(self.device)
                msk_x = msk_x.to(self.device)
                msk_y = msk_y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                inst = inst.to(self.device)
                with torch.autograd.set_detect_anomaly(True):
                    with torch.set_grad_enabled(is_train):
                        leap_loss, gt_traj_energy, mcmc_energy, \
                            action_correct_rate, action_correct_rate_steps, pos_energy, neg_energy \
                            = leap_model.train_step(x, y, full_imgs, state_masks=msk_x, \
                                timesteps=t, insts=inst, init_states=init_x, init_obss=init_image, rtgs=r, logger = self.logger, task_name = self.env_name) 
                        if (leap_loss == 0):
                            continue
                        leap_loss = leap_loss.mean() 
                        leap_losses.append(leap_loss.item())
                        gt_traj_energys.append(gt_traj_energy)
                        mcmc_energys.append(mcmc_energy)
                        action_correct_rates.append(action_correct_rate)
                        all_action_correct_rate_steps.append(action_correct_rate_steps)
                        pos_energies.append(pos_energy.item())
                        neg_energies.append(neg_energy.item())

                    leap_model.zero_grad()
                    leap_loss.backward()
                    torch.nn.utils.clip_grad_norm_(leap_model.parameters(), config.grad_norm_clip)
                    leap_optimizer.step()
                
                if is_train:
                    # backprop and update the parameters in model
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in leap_optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    #pbar.set_description(f"epoch {epoch+1} iter {it}: leap loss {leap_loss.item():.5f}. lr {lr:e}")
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):#config.max_epochs
            run_epoch('train', epoch_num=epoch)  
            #print(f"epoch {epoch}")
            #if (epoch % 100 == 0): # and epoch > 90
            #    self.test_returns(0, test_num=20) 
            #    print(epoch)
        #print("start testing")
        #self.save_checkpoint()
        self.test_returns(0, self.env_name, self.plan_horizon, test_num=40) 
        

    def test_returns(self, ret, env_name, plan_horizon, test_num=40):
        self.leap_model.train(False)
        env = self.env
        T_rewards = []
        done = True
        success_count = 0
        for i in range(test_num):
            obs = env.reset()
            self.bot_advisor_agent = BotAgent(self.env)
            goals = self.bot_advisor_agent.get_goal_state()
            goal = self.goal_selection(goals, env.gen_agent_pos())

            reward_sum = 0
            done = False
            full_obs = env.gen_full_obs()
            full_obs = torch.from_numpy(full_obs).flatten().to(dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            insts = torch.tensor(self.inst_preprocessor(obs['mission']), dtype=torch.long).unsqueeze(0).to(self.device)
            state = env.gen_agent_pos() + [obs['direction']] + goal
            state = torch.Tensor(state).type(torch.long).unsqueeze(0).unsqueeze(0).to(self.device)

            # first state is from env, first rtg is target return, and first timestep is 0
            sample_actions = leap_sample_multi_step(self.leap_model, env_name, plan_horizon, state, insts=insts, full_obs=full_obs, \
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), logger=self.logger, sample_iteration=self.sample_iteration)

            all_states = state
            all_full_obs = full_obs
            actions = []
            j = 0
            
            while True:
                stuck_action = -1
                for action in sample_actions: #[:1]
                    # move forward
                    last_state = env.gen_agent_pos()
                    obs, reward, done, info = env.step(action)
                    actions += [action]
                    reward_sum += reward
                    if done:  
                        break
                    j = len(actions)
                    cur_state = env.gen_agent_pos()
                    state = cur_state + [obs['direction']] + goal
                    state = torch.Tensor(state).type(torch.long).unsqueeze(0).unsqueeze(0).to(self.device)
                    full_obs = env.gen_full_obs()
                    full_obs = torch.from_numpy(full_obs).flatten().to(dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                    all_states = torch.cat([all_states, state], dim=1)
                    all_full_obs = torch.cat([all_full_obs, full_obs], dim=1)
                if (done):
                    T_rewards.append(reward_sum)
                    # print(f"actions {actions}, length {len(actions)}") #, all_states {all_states}
                    # succeed if reward_sum is positive, timeout reward_sum is zero
                    if (reward_sum > 0):
                        success_count += 1
                    break
                sample_actions = leap_sample_multi_step(self.leap_model, env_name, plan_horizon, all_states, insts=insts, full_obs=all_full_obs, logger=self.logger,\
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)), \
                    sample_iteration=self.sample_iteration)

        env.close()
        eval_return = sum(T_rewards)/float(test_num)
        success_rate = success_count /float(test_num)
        msg = f"eval return: {eval_return}, success_rate: {success_rate:.3f}"
        self.logger.info(msg)
        self.leap_model.train(True)
        return eval_return

    def goal_selection(self, goals, agent):
        min_dist = 100
        min_idx = -1
        for i, goal in enumerate(goals):
            dist = np.sqrt((agent[0]-goal[0])**2+(agent[1]-goal[1])**2)
            if (dist < min_dist):
                min_dist = dist
                min_idx = i
        return list(goals[min_idx])


    def search_turn_direction(self, goal_direction, current_direction):
        if (goal_direction == 0):
            if (current_direction in [2,3]):
                return 1
            elif (current_direction == 1):
                return 0
        elif (goal_direction == 1):
            if (current_direction in [0,3]):
                return 1
            elif (current_direction == 2):
                return 0
        elif (goal_direction == 2):
            if (current_direction in [0,1]):
                return 1
            elif (current_direction == 3):
                return 0
        elif (goal_direction == 3):
            if (current_direction in [2,1]):
                return 1
            elif (current_direction == 0):
                return 0