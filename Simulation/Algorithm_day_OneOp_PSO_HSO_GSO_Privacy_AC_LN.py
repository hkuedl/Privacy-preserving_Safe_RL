import numpy as np
import torch
import copy
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cProfile
matplotlib.use('Agg')

import sys
# sys.path.append("workspaces/Multi_energy/Simulation")
from Simulation.Correction_Module_PSO_HSO_GSOV2_LN_Simple import Correction_Module



# from Algorithm.torch_utils.torch_utils import get_device
# import math
# from utils.distributions.categorical import Categorical


class BatchRLAlgorithm:
    def __init__(self, trainer, env, policy, replay_buffer, num_epochs, num_env_steps_per_epoch, num_trains_per_train_epoch, min_num_epoch_before_training, batch_size, result_dir, **env_args):
        self.env = env
        self.trainer = trainer
        self.policy = policy
        self.max_epoch_num = num_epochs
        self.max_env_step = num_env_steps_per_epoch
        self.num_trains_per_train_epoch = num_trains_per_train_epoch
        self.min_num_epoch_before_training = min_num_epoch_before_training
        self.batch_size = batch_size
        self.obs_dim = self.env.get_obs_dim()
        self.action_dim = self.env.get_action_dim()
        self.replay_buffer = replay_buffer
        self.device_config = self.env.get_device_config()
        self.result_dir = result_dir



    def train(self):
        obs, obs_encry, done = self.env.reset()
        start_time = time.time()
        reward_curve = []
        cost_curve = []
        #  Check the algorithm
        HO = []
        HO_en = []
        HSAC = []
        HA = []
        T = []
        OV = []

        update_times = self.num_trains_per_train_epoch

        for self.epoch in tqdm(range(self.max_epoch_num)):
            if self.epoch < self.min_num_epoch_before_training:
                obs, obs_encry, done = self.env.reset_SOC(self.epoch)
                obs, done, episodic_reward, ho, ho_en, hsac, ha, ht, objVal, episodic_cost = self.run_sim(self.epoch, obs, obs_encry)
            else:
                obs, obs_encry, done = self.env.reset_SOC(self.epoch)
                obs, done, episodic_reward, ho, ho_en, hsac, ha, ht, objVal, episodic_cost = self.run_sim(self.epoch, obs, obs_encry)
                for _ in range(update_times):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
            
            k = 1 + (self.replay_buffer.num_steps_can_sample() / self.replay_buffer.replay_buffer_size())
            update_times = int(k * self.num_trains_per_train_epoch)

            HO += ho
            HO_en += ho_en
            HSAC += hsac
            HA += ha
            T += ht
            OV += objVal 

            reward_curve.append(episodic_reward)
            cost_curve.append(episodic_cost)
            if len(reward_curve) % 100 == 0 and len(reward_curve) != 0:
                plt.figure(figsize=(20,10))
                x_plot = np.arange(len(reward_curve))
                plt.plot(x_plot, np.array(reward_curve), linewidth=2, color="orange")  
                plt.xlabel('Episode', fontsize =14)
                plt.ylabel('Reawrd', fontsize =14)
                plt.grid()
                plt.savefig(self.result_dir+'/Reward.png')
                plt.show()
                plt.close()

                plt.figure(figsize=(20,10))
                x_plot = np.arange(len(cost_curve))
                plt.plot(x_plot, np.array(cost_curve), linewidth=2, color="orange")  
                plt.xlabel('Episode', fontsize =14)
                plt.ylabel('Cost', fontsize =14)
                plt.grid()
                plt.savefig(self.result_dir+'/Cost.png')
                plt.show()
                plt.close()

            if len(reward_curve) % 500 == 0 and len(reward_curve) != 0:
                np.save(self.result_dir+'/Cost.npy', np.array(reward_curve))
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # if self.epoch % 500 == 0: 
            #     print("\n", f"Elapsed Time: {int(hours)}h/{int(minutes)}min/{seconds:.2f}sec", "Epoch:", self.epoch, "Violation:", np.array(objVal).sum(), "Episodic reward:", episodic_reward, "Episodic cost:", episodic_cost)
        np.save(self.result_dir+'/his_obs.npy', np.array(HO))
        np.save(self.result_dir+'/his_obs_encry.npy', np.array(HO_en))
        np.save(self.result_dir+'/his_sac.npy', np.array(HSAC))
        np.save(self.result_dir+'/his_action.npy', np.array(HA))
        np.save(self.result_dir+'/his_trade.npy', np.array(T))
        np.save(self.result_dir+'/his_OV.npy', np.array(OV))
        np.save(self.result_dir+'/Reward.npy', np.array(reward_curve))
        np.save(self.result_dir+'/Cost.npy', np.array(cost_curve))
        
        # 保存policy模型的状态字典
        torch.save(self.policy.state_dict(), self.result_dir+'/policy.pth')


    def run_sim(self, epoch, obs=None, obs_encry=None):

        o = obs
        o_en = obs_encry
        cum_r = 0
        cum_cost = 0
        his_a_sac = []
        his_action = []
        his_obs = []
        his_obs_encry = []
        his_trade = []
        objVal = []

        for env_step in range(self.max_env_step):
            state = o
            state_encry = o_en
            a_sac, _ = self.policy.get_action(state_encry)
            len = self.env.get_action_dim() + 3
            a = np.zeros(len)                  

            # Action Rate Formulation 
            for action, i in zip(a_sac, range(18)):       #除了储能外，其他action的范围为[0,1]，而policy输出的action为[-1,1],因此需要处理一下再输出。
                if i == 8 or i == 9 or i == 10 or i == 15 or i == 16 or i == 17:
                    a[i] = action
                else:
                    # a[i] = max(action, 0)                         #截断处理
                    a[i] = (action + 1)/2                          #转换到[0,1]

            # HSO action                    (a_asc[8]给出的是EHP和GB分别承担的负荷比例，a_sac[9]给出的是TS的动作)
            a[18] = (a_sac[18] + 1)/2
            a[19] = (a_sac[19] + 1)/2
            a[20] = (a_sac[20] + 1)/2
            a[21] = 1 - a[18]
            a[22] = 1 - a[19]
            a[23] = 1 - a[20]
            a[24] = a_sac[21]
            a[25] = a_sac[22]
            a[26] = a_sac[23]

            ## correction module for action
            load = self.env.get_load(epoch, env_step)
            correction = Correction_Module(state, a, self.device_config, load)
            a, trade, OV = correction.get_action()
            
            next_o, next_o_encry, r, cost, done = self.env.step(epoch, copy.deepcopy(a), copy.deepcopy(trade), env_step, self.max_env_step)

            for action, i in zip(a, range(len)):           # 将action转换为policy输出的形式记录，用于更新
                if i == 8 or i == 9 or i == 10 or i == 15 or i == 16 or i == 17 or i == 24 or i == 25 or i == 26:
                    continue
                else:
                    a[i] = action * 2 - 1

            # action = a               # 存储安全action
            action = a_sac             # 存储原action
            reward = r
            terminal = done
            next_state = next_o_encry
            self.replay_buffer.add_sample(state_encry, action, reward, next_state, terminal)
            
            cum_r += r
            cum_cost += cost
            objVal.append(OV)

            # check the algorithm
            his_a_sac.append(a_sac)
            his_action.append(a)
            his_obs.append(o)
            his_obs_encry.append(o_en)
            his_trade.append(trade)

            if done:
                break
            o = next_o
            o_en = next_o_encry
            
        return o, done, cum_r, his_obs, his_obs_encry, his_a_sac, his_action, his_trade, objVal, cum_cost

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


