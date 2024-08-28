import numpy as np
import torch
import copy
import datetime
import matplotlib.pyplot as plt

import sys
# sys.path.append("workspaces/Multi_energy/Simulation")
from Simulation.Correction_Module_PSO_HSO_GSOV2_LN_Simple import Correction_Module



# from Algorithm.torch_utils.torch_utils import get_device
# import math
# from utils.distributions.categorical import Categorical


class BatchRLAlgorithm:
    def __init__(self, env, policy, num_env_steps_per_epoch, **env_args):
        self.env = env
        self.policy = policy
        self.max_env_step = num_env_steps_per_epoch
        self.obs_dim = self.env.get_obs_dim()
        self.action_dim = self.env.get_action_dim()
        self.device_config = self.env.get_device_config()

    def test(self):
        obs, obs_encry, done = self.env.reset()
        reward_curve = []
        cost_curve = []
        #  Check the algorithm
        HO = []
        HO_en = []
        HSAC = []
        HA = []
        T = []
        OV = []

        #  Check the constraints
        PSO_P = [] 
        PSO_Q  = [] 
        VOL = [] 
        GSO_F = [] 
        PRES = [] 
        TEMP_S = [] 
        TEMP_R = [] 


        for self.epoch in range(334,365):

            obs, obs_encry, done = self.env.reset_SOC(self.epoch)
            obs, done, episodic_reward, ho, ho_en, hsac, ha, ht, objVal, episodic_cost, \
            pso_flow_p, pso_flow_q, Voltage, gso_flow, Pressure, temp_supply, temp_return = self.run_sim(self.epoch, obs, obs_encry)
            print("Epoch:", self.epoch, "Violation:", np.array(objVal).sum(), "Episodic reward:", episodic_reward, "Episodic cost:", episodic_cost)


            HO += ho
            HO_en += ho_en
            HSAC += hsac
            HA += ha
            T += ht
            OV += objVal 

            #  Check the constraints
            PSO_P += pso_flow_p
            PSO_Q += pso_flow_q
            VOL += Voltage
            GSO_F += gso_flow
            PRES += Pressure
            TEMP_S += temp_supply
            TEMP_R += temp_return


            reward_curve.append(episodic_reward)
            cost_curve.append(episodic_cost)
            
        np.save('workspaces/Multi_energy/Results/Test_LN/his_obs.npy', np.array(HO))
        np.save('workspaces/Multi_energy/Results/Test_LN/his_obs_encry.npy', np.array(HO_en))
        np.save('workspaces/Multi_energy/Results/Test_LN/his_sac.npy', np.array(HSAC))
        np.save('workspaces/Multi_energy/Results/Test_LN/his_action.npy', np.array(HA))
        np.save('workspaces/Multi_energy/Results/Test_LN/his_trade.npy', np.array(T))
        np.save('workspaces/Multi_energy/Results/Test_LN/his_OV.npy', np.array(OV))
        np.save('workspaces/Multi_energy/Results/Test_LN/Reward.npy', np.array(reward_curve))
        np.save('workspaces/Multi_energy/Results/Test_LN/Cost.npy', -np.array(cost_curve)/1000)
        np.save('workspaces/Multi_energy/Results/Test_LN/pso_flow_p.npy', np.array(PSO_P))
        np.save('workspaces/Multi_energy/Results/Test_LN/pso_flow_q.npy', np.array(PSO_Q))
        np.save('workspaces/Multi_energy/Results/Test_LN/Voltage.npy', np.array(VOL))
        np.save('workspaces/Multi_energy/Results/Test_LN/gso_flow.npy', np.array(GSO_F))
        np.save('workspaces/Multi_energy/Results/Test_LN/Pressure.npy', np.array(PRES))
        np.save('workspaces/Multi_energy/Results/Test_LN/temp_supply.npy', np.array(TEMP_S))
        np.save('workspaces/Multi_energy/Results/Test_LN/temp_return.npy', np.array(TEMP_R))
        print('cumulative cost:', -(np.array(cost_curve).sum())/1000)
        return -(np.array(cost_curve).sum())/1000
               


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

        pso_flow_p = [] 
        pso_flow_q  = [] 
        Voltage = [] 
        gso_flow = [] 
        Pressure = [] 
        temp_supply = [] 
        temp_return = [] 


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
            correction = Correction_Module(state, a, self.device_config, load, test=True)
            a, trade, OV, pso_p, pso_q, Vol, gso_f, Pres, temp_sup, temp_re = correction.get_action()

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
            
            cum_r += r
            cum_cost += cost
            objVal.append(OV)

            # check the algorithm
            his_a_sac.append(a_sac)
            his_action.append(a)
            his_obs.append(o)
            his_obs_encry.append(o_en)
            his_trade.append(trade)

            # check the constraints
            pso_flow_p.append(pso_p)
            pso_flow_q.append(pso_q)
            Voltage.append(Vol)
            gso_flow.append(gso_f)
            Pressure.append(Pres)
            temp_supply.append(temp_sup)
            temp_return.append(temp_re)
            
            if done:
                break
            o = next_o
            o_en = next_o_encry
             
        return o, done, cum_r, his_obs, his_obs_encry, his_a_sac, his_action, his_trade, objVal, cum_cost, pso_flow_p, pso_flow_q, Voltage, gso_flow, Pressure, temp_supply, temp_return

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


