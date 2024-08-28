import gym
from gym import spaces
import numpy as np
import pandas as pd
import math
import copy


class Multi_Energy_Env(gym.Env):
    """
    Observation(PSO):
        Type: -
        Num     Observation                                    Min                     Max
        0-2     SoC of energy storage 1 (3dim)                  -                       -
        3       Real-time electricity price(buy)                -                       -
        4       Real-time electricity price(sell)               -                       -
        5-9     Predicted output power of PV (5dim)             -                       -
        10-12   Predicted output power of wind turbine (3dim)   -                       -
        13      Energy demand (active power)                    -                       -
        14      Energy demand (reactive power)                  -                       -

    Observation(GSO):
        Type: -
        Num     Observation                                    Min                     Max
        15-17   SoC of gas storage (3dim)                       -                       -
        18-28   Gas demand (11 dim)                             -                       -
       
    Observation(HSO):
        Type: -
        Num     Observation                                    Min                     Max
        29-31   SoC of thermal storage (3dim)                   -                       -
        32      Heat demand                                     -                       -
    
    Observation_general:  (为了方便管理,放到了HSO的obs里面)                                 
        33      Time                                            -                       - 

    Actions(PSO):
        Type: -
        Num       Action                                        Min                    Max
        0         Output of diesel generator 1                   0                      1
        1         Output of diesel generator 2                   0                      1
        2         Output of gas turbine 1                        0                      1
        3         Output of gas turbine 2                        0                      1
        4         Output of diesel generator 1 (reactive power)  0                      1
        5         Output of diesel generator 2 (reactive power)  0                      1
        6         Output of gas turbine 1 (reactive power)       0                      1
        7         Output of gas turbine 2 (reactive power)       0                      1
        8         Charging/discharging of energy storage 1      -1                      1
        9         Charging/discharging of energy storage 2      -1                      1
        10        Charging/discharging of energy storage 3      -1                      1

    Actions(GSO):
        Type: -
        Num       Action                                        Min                    Max
        11         Output of gas well 1                           0                      1
        12         Output of gas well 2                           0                      1
        13         Output of gas well 3                           0                      1
        14         Output of gas well 4                           0                      1
        15         Charging/discharging of gas storage 1         -1                      1
        16         Charging/discharging of gas storage 2         -1                      1
        17         Charging/discharging of gas storage 3         -1                      1
        
    Actions(HSO):
        Type: -
        Num       Action                                        Min                    Max
        18         Output of electric heat pump 1                 0                      1
        19         Output of electric heat pump 2                 0                      1
        20         Output of electric heat pump 3                 0                      1
        21         Output of gas boiler 1                         0                      1
        22         Output of gas boiler 2                         0                      1
        23         Output of gas boiler 3                         0                      1
        24        Charging/discharging of thermal storage 1      -1                      1
        25        Charging/discharging of thermal storage 2      -1                      1
        26        Charging/discharging of thermal storage 3      -1                      1
    """

    def __init__(self):
        
        self.oneday_flag = False                       # 选择使用一天的数据来训练（True）还是使用11个月的数据来训练(False)。

        ## Parameters setting
        self.rated_power_factor = 0.85     # 用于折算无功负荷
        self.gamma = 0.9                   # 用于计算reward
        self.operator_name = ['pso', 'gso', 'hso']    #运营商名称
        self.obs_num = [15, 14, 5]                      #各运营商observation数量
        self.action_dim = 24
        self.obs_dim = 34
        self.seed = 0
        ### parameters of storages
        self.es_capacity = 200                         # energy capacity of energy storage
        self.gs_capacity = 250                         # energy capacity of gas storage
        self.ts_capacity = 400                         # energy capacity of thermal storage
        self.es_efficiency = 0.90                      #充放效率
        self.gs_efficiency = 0.95
        self.ts_efficiency = 0.90
        ### power max of devices
        self.dg_power_max = 600                        # max power out of diesel generator
        self.gt_power_max = 600                        # max power out of gas turbine
        self.dg_q_power_max = 200                       # max reactive power out of diesel generator
        self.gt_q_power_max = 200                       # max reactive power out of gas turbine
        self.gb_power_max = 200                        # max power out of gas boiler
        self.ehp_power_max = 200                       # max power out of electric heat pump
        self.gw1_power_max = 500                       # max power out of gas well 1
        self.gw2_power_max = 500                       # max power out of gas well 2
        self.es_power_max = 50                         # max power out of energy storage
        self.gs_power_max = 60                         # max power out of gas storage
        self.ts_power_max = 50                         # max power out of heat storage
        ### production cost of devices
        self.cost_dg = 0.210  # 维护成本+可变成本（燃料）
        self.cost_gt = 0.080
        self.cost_dg_Q = 0.021  # 维护成本+可变成本（燃料）
        self.cost_gt_Q = 0.018
        self.cost_gw1 = 0.20
        self.cost_gw2 = 0.20
        self.cost_gb = 0.05   
        self.cost_ehp = 0.04
        ### carbon emission rate of devices
        self.cer_dg = 0.000442      # t/kWh
        self.cer_gt = 0.000368     # t/kWh
        self.cer_gb = 0.000368     # t/kWh
        carbon_price = 50          # 50 pound / t

        ## Data loading
        energy_demand = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 0)
        energy_demand_q = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 6)
        heat_demand = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 1)
        gas_demand = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 2)
        elec_price = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 3)
        gas_price = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 4)
        res = pd.read_excel('workspaces/Multi_energy/Environment/dataset.xlsx', sheet_name = 5)
        
        ### Demand
        self.energy_demand = np.array(energy_demand).reshape((365,24,3))
        # self.energy_demand_q = np.array(energy_demand_q).reshape((365,24,3))
        self.energy_demand_q = self.energy_demand * np.tan(np.arccos(0.95))
        self.gas_demand = np.array(gas_demand)
        self.heat_demand = np.array(heat_demand).reshape((365,24,4))


        ### Price
        self.electricity_price = np.array(elec_price)
        # self.gas_price = np.array(gas_price)
        self.gas_price = 0.20
        self.carbon_price = carbon_price
        ### Renewable energy output
        res = np.array(res)
        self.wind = res[:,0].reshape((365,24))
        self.pv = res[:,1].reshape((365,24))

        ## Initialize states of each operators (输入env_step = 0的数据，并且初始化三类储能的SOC)
        initialized_episode_step = 0
        initialized_env_step = 0
        initial_states = {}
        for i,j in zip(self.operator_name, self.obs_num):
            initial_states[i] = np.zeros((j))

        ### PSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[0]][0] = 0
        initial_states[self.operator_name[0]][1] = 0
        initial_states[self.operator_name[0]][2] = 0
        initial_states[self.operator_name[0]][3] = self.electricity_price[initialized_env_step]               
        initial_states[self.operator_name[0]][4] = self.electricity_price[initialized_env_step] / 2         #购价为售价的2倍
        initial_states[self.operator_name[0]][5] = self.pv[initialized_episode_step, initialized_env_step] * 0.9
        initial_states[self.operator_name[0]][6] = self.pv[initialized_episode_step, initialized_env_step] * 1.0
        initial_states[self.operator_name[0]][7] = self.pv[initialized_episode_step, initialized_env_step] * 1.1
        initial_states[self.operator_name[0]][8] = self.pv[initialized_episode_step, initialized_env_step] * 1.2
        initial_states[self.operator_name[0]][9] = self.pv[initialized_episode_step, initialized_env_step] * 1.3
        initial_states[self.operator_name[0]][10] = self.wind[initialized_episode_step, initialized_env_step] * 1.0
        initial_states[self.operator_name[0]][11] = self.wind[initialized_episode_step, initialized_env_step] * 1.1
        initial_states[self.operator_name[0]][12] = self.wind[initialized_episode_step, initialized_env_step] * 1.2
        initial_states[self.operator_name[0]][13] = self.energy_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[0]][14] = self.energy_demand_q[initialized_episode_step, initialized_env_step].sum()

        ### GSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[1]][0] = 0
        initial_states[self.operator_name[1]][1] = 0
        initial_states[self.operator_name[1]][2] = 0
        # initial_states[self.operator_name[0]][1] = self.gas_demand[initialized_env_step].sum()
        initial_states[self.operator_name[1]][3] = self.gas_demand[initialized_env_step][0]
        initial_states[self.operator_name[1]][4] = self.gas_demand[initialized_env_step][0]
        initial_states[self.operator_name[1]][5] = self.gas_demand[initialized_env_step][1]
        initial_states[self.operator_name[1]][6] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][7] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][8] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][9] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][10] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][11] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][12] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][13] = self.gas_demand[initialized_env_step][1]


        ### HSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[2]][0] = 0
        initial_states[self.operator_name[2]][1] = 0
        initial_states[self.operator_name[2]][2] = 0
        initial_states[self.operator_name[2]][3] = self.heat_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[2]][4] = 0

        self.states = initial_states


    def step(self, epoch, action, trade, env_step, max_env_step):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        action = action
        
        iteration = epoch
        iteration_hso = iteration

        ## output of devices
        output_dg1 = action[0] * self.dg_power_max
        output_dg2 = action[1] * self.dg_power_max
        output_gt1 = action[2] * self.gt_power_max
        output_gt2 = action[3] * self.gt_power_max
        output_dg_q1 = action[4] * self.dg_q_power_max
        output_dg_q2 = action[5] * self.dg_q_power_max
        output_gt_q1 = action[6] * self.gt_q_power_max 
        output_gt_q2 = action[7] * self.gt_q_power_max        
        output_es1 = action[8] * self.es_power_max
        output_es2 = action[9] * self.es_power_max
        output_es3 = action[10] * self.es_power_max

        output_gw1 = action[11] * self.gw1_power_max
        output_gw2 = action[12] * self.gw1_power_max
        output_gw3 = action[13] * self.gw2_power_max
        output_gw4 = action[14] * self.gw2_power_max
        output_gs1 = action[15] * self.gs_power_max
        output_gs2 = action[16] * self.gs_power_max
        output_gs3 = action[17] * self.gs_power_max

        output_ehp1 = action[18] * self.ehp_power_max
        output_ehp2 = action[19] * self.ehp_power_max
        output_ehp3 = action[20] * self.ehp_power_max
        output_gb1 = action[21] * self.gb_power_max
        output_gb2 = action[22] * self.gb_power_max
        output_gb3 = action[23] * self.gb_power_max
        output_ts1 = action[24] * self.ts_power_max
        output_ts2 = action[25] * self.ts_power_max
        output_ts3 = action[26] * self.ts_power_max

        ##State transition
        
        # if iteration == 333 and env_step >= max_env_step - 1:             # 前11个月为一个轮次
        #     done = True
        # else:
        #     done = False

        if env_step >= max_env_step - 1:                                    # 每天都为一个轮次
            done = True
        else:
            done = False

        ### PSO state transition
        if output_es1 >= 0:                          #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[0]][0] += output_es1 / self.es_capacity * self.es_efficiency     #delta t 默认为1h, 因为action是肯定保证不会让储能SOC超出范围[0,1]，因此这里不需要约束
        else:
            self.states[self.operator_name[0]][0] += output_es1 / self.es_capacity / self.es_efficiency
        
        if output_es2 >= 0:                          #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[0]][1] += output_es2 / self.es_capacity * self.es_efficiency     #delta t 默认为1h, 因为action是肯定保证不会让储能SOC超出范围[0,1]，因此这里不需要约束
        else:
            self.states[self.operator_name[0]][1] += output_es2 / self.es_capacity / self.es_efficiency

        if output_es3 >= 0:                          #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[0]][2] += output_es3 / self.es_capacity * self.es_efficiency     #delta t 默认为1h, 因为action是肯定保证不会让储能SOC超出范围[0,1]，因此这里不需要约束
        else:
            self.states[self.operator_name[0]][2] += output_es3 / self.es_capacity / self.es_efficiency
        
        if not done:
            self.states[self.operator_name[0]][3] = self.electricity_price[env_step+1] 
            self.states[self.operator_name[0]][4] = self.electricity_price[env_step+1] / 2
            self.states[self.operator_name[0]][5] = self.pv[iteration][env_step+1] * 0.9
            self.states[self.operator_name[0]][6] = self.pv[iteration][env_step+1] * 1.0
            self.states[self.operator_name[0]][7] = self.pv[iteration][env_step+1] * 1.1
            self.states[self.operator_name[0]][8] = self.pv[iteration][env_step+1] * 1.2
            self.states[self.operator_name[0]][9] = self.pv[iteration][env_step+1] * 1.3
            self.states[self.operator_name[0]][10] = self.wind[iteration][env_step+1] * 1.0
            self.states[self.operator_name[0]][11] = self.wind[iteration][env_step+1] * 1.1
            self.states[self.operator_name[0]][12] = self.wind[iteration][env_step+1] * 1.2
            self.states[self.operator_name[0]][13] = self.energy_demand[iteration][env_step+1].sum()
            self.states[self.operator_name[0]][14] = self.energy_demand_q[iteration][env_step+1].sum()
        else:
            self.states[self.operator_name[0]][3] = 0
            self.states[self.operator_name[0]][4] = 0
            self.states[self.operator_name[0]][5] = 0
            self.states[self.operator_name[0]][6] = 0
            self.states[self.operator_name[0]][7] = 0
            self.states[self.operator_name[0]][8] = 0
            self.states[self.operator_name[0]][9] = 0
            self.states[self.operator_name[0]][10] = 0
            self.states[self.operator_name[0]][11] = 0
            self.states[self.operator_name[0]][12] = 0
            self.states[self.operator_name[0]][13] = 0
            self.states[self.operator_name[0]][14] = 0
        
        ### GSO state transition
        if output_gs1 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[1]][0] += output_gs1 / self.gs_capacity * self.gs_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[1]][0] += output_gs1 / self.gs_capacity / self.gs_efficiency
        if output_gs2 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[1]][1] += output_gs2 / self.gs_capacity * self.gs_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[1]][1] += output_gs2 / self.gs_capacity / self.gs_efficiency
        if output_gs3 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[1]][2] += output_gs3 / self.gs_capacity * self.gs_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[1]][2] += output_gs3 / self.gs_capacity / self.gs_efficiency
        
        if not done:
            # self.states[self.operator_name[1]][1] = self.gas_demand[env_step+1].sum()
            self.states[self.operator_name[1]][3] = self.gas_demand[env_step+1][0]
            self.states[self.operator_name[1]][4] = self.gas_demand[env_step+1][0]
            self.states[self.operator_name[1]][5] = self.gas_demand[env_step+1][1]
            self.states[self.operator_name[1]][6] = self.gas_demand[env_step+1][3]
            self.states[self.operator_name[1]][7] = self.gas_demand[env_step+1][2]
            self.states[self.operator_name[1]][8] = self.gas_demand[env_step+1][3]
            self.states[self.operator_name[1]][9] = self.gas_demand[env_step+1][2]
            self.states[self.operator_name[1]][10] = self.gas_demand[env_step+1][2]
            self.states[self.operator_name[1]][11] = self.gas_demand[env_step+1][3]
            self.states[self.operator_name[1]][12] = self.gas_demand[env_step+1][2]
            self.states[self.operator_name[1]][13] = self.gas_demand[env_step+1][1]
        else:
            # self.states[self.operator_name[1]][1] = 0
            self.states[self.operator_name[1]][3] = 0
            self.states[self.operator_name[1]][4] = 0
            self.states[self.operator_name[1]][5] = 0
            self.states[self.operator_name[1]][6] = 0
            self.states[self.operator_name[1]][7] = 0
            self.states[self.operator_name[1]][8] = 0
            self.states[self.operator_name[1]][9] = 0
            self.states[self.operator_name[1]][10] = 0
            self.states[self.operator_name[1]][11] = 0
            self.states[self.operator_name[1]][12] = 0
            self.states[self.operator_name[1]][13] = 0

        ### HSO state transition
        if output_ts1 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[2]][0] += output_ts1 / self.ts_capacity * self.ts_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[2]][0] += output_ts1 / self.ts_capacity / self.ts_efficiency

        if output_ts2 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[2]][1] += output_ts2 / self.ts_capacity * self.ts_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[2]][1] += output_ts2 / self.ts_capacity / self.ts_efficiency

        if output_ts3 >= 0:                              #判断充放状态: 正值为充能，负值为放能
            self.states[self.operator_name[2]][2] += output_ts3 / self.ts_capacity * self.ts_efficiency     #delta t 默认为1h
        else:
            self.states[self.operator_name[2]][2] += output_ts3 / self.ts_capacity / self.ts_efficiency

        if not done:
            self.states[self.operator_name[2]][3] = self.heat_demand[iteration_hso][env_step+1].sum()
        else:
            self.states[self.operator_name[2]][3] = 0
        self.states[self.operator_name[2]][4] = env_step + 1

        ## Reward calculation

        ## 计算电网-电主网和气网-气主网交互的电量与天然气量
        energy_delta = trade[0]
        energy_delta_q = trade[1]

        lambda_gas = self.gas_price
        lambda_elc = self.electricity_price[env_step]

        if energy_delta >= 0:
            lambda_electricity = self.electricity_price[env_step]
        else:
            lambda_electricity = self.electricity_price[env_step] / 2

        ### PSO reward & cost
        reward_pso_p = - (lambda_electricity * energy_delta) - (self.cost_dg * output_dg1) - (self.cost_dg * output_dg2) - (self.cost_gt * output_gt1) - (self.cost_gt * output_gt2)
        reward_pso_q = - (lambda_electricity * energy_delta_q) - (self.cost_dg_Q * output_dg_q1) - (self.cost_dg_Q * output_dg_q2) - (self.cost_gt_Q * output_gt_q1) - (self.cost_gt_Q * output_gt_q2)

        reward_pso = reward_pso_p + reward_pso_q*3

        cost_pso_q = - (0.1*lambda_electricity * energy_delta_q) - (self.cost_dg_Q * output_dg_q1) - (self.cost_dg_Q * output_dg_q2) - (self.cost_gt_Q * output_gt_q1) - (self.cost_gt_Q * output_gt_q2)
        cost_pso = - (lambda_electricity * energy_delta) - (self.cost_dg * output_dg1) - (self.cost_dg * output_dg2) - (self.cost_gt * output_gt1) - (self.cost_gt * output_gt2)

        ### GSO reward & cost
        gas_delta = trade[2]

        reward_gso = - (lambda_gas * gas_delta) - (self.cost_gw1  * output_gw1) - (self.cost_gw1 * output_gw2) - (self.cost_gw2  * output_gw3) - (self.cost_gw2 * output_gw4)

        cost_gso = - (lambda_gas * gas_delta) - (self.cost_gw1  * output_gw1) - (self.cost_gw1 * output_gw2) - (self.cost_gw2  * output_gw3) - (self.cost_gw2 * output_gw4)

        ### HSO reward & cost

        reward_hso = - (self.cost_gb * output_gb1) - (self.cost_ehp  * output_ehp1) - (self.cost_gb * output_gb2) - (self.cost_ehp  * output_ehp2) - (self.cost_gb * output_gb3) - (self.cost_ehp  * output_ehp3) 
    
        cost_hso = - (self.cost_gb * output_gb1) - (self.cost_ehp  * output_ehp1) - (self.cost_gb * output_gb2) - (self.cost_ehp  * output_ehp2) - (self.cost_gb * output_gb3) - (self.cost_ehp  * output_ehp3)

        # summation
        reward = (reward_pso + reward_gso + reward_hso) * 0.2
        cost = cost_pso + cost_gso + cost_hso

        state = self.flatten_states(self.states)                          #把存储state的字典，转换为展平的一维array

        return state, self.IM(copy.deepcopy(state)), reward, cost, done

    def reset(self):
        ## Initialize states of each operators (输入env_step = 0的数据，并且初始化三类储能的SOC)
        initialized_episode_step = 0
        initialized_env_step = 0
        initial_states = {}
        for i,j in zip(self.operator_name, self.obs_num):
            initial_states[i] = np.zeros((j))

        ### PSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[0]][0] = 0
        initial_states[self.operator_name[0]][1] = 0
        initial_states[self.operator_name[0]][2] = 0
        initial_states[self.operator_name[0]][3] = self.electricity_price[initialized_env_step]               
        initial_states[self.operator_name[0]][4] = self.electricity_price[initialized_env_step] / 2         #购价为售价的2倍
        initial_states[self.operator_name[0]][5] = self.pv[initialized_episode_step, initialized_env_step] * 0.9
        initial_states[self.operator_name[0]][6] = self.pv[initialized_episode_step, initialized_env_step] * 1.0
        initial_states[self.operator_name[0]][7] = self.pv[initialized_episode_step, initialized_env_step] * 1.1
        initial_states[self.operator_name[0]][8] = self.pv[initialized_episode_step, initialized_env_step] * 1.2
        initial_states[self.operator_name[0]][9] = self.pv[initialized_episode_step, initialized_env_step] * 1.3
        initial_states[self.operator_name[0]][10] = self.wind[initialized_episode_step, initialized_env_step] * 1.0
        initial_states[self.operator_name[0]][11] = self.wind[initialized_episode_step, initialized_env_step] * 1.1
        initial_states[self.operator_name[0]][12] = self.wind[initialized_episode_step, initialized_env_step] * 1.2
        initial_states[self.operator_name[0]][13] = self.energy_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[0]][14] = self.energy_demand_q[initialized_episode_step, initialized_env_step].sum()

        ### GSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[1]][0] = 0
        initial_states[self.operator_name[1]][1] = 0
        initial_states[self.operator_name[1]][2] = 0
        # initial_states[self.operator_name[0]][1] = self.gas_demand[initialized_env_step].sum()
        initial_states[self.operator_name[1]][3] = self.gas_demand[initialized_env_step][0]
        initial_states[self.operator_name[1]][4] = self.gas_demand[initialized_env_step][0]
        initial_states[self.operator_name[1]][5] = self.gas_demand[initialized_env_step][1]
        initial_states[self.operator_name[1]][6] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][7] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][8] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][9] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][10] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][11] = self.gas_demand[initialized_env_step][3]
        initial_states[self.operator_name[1]][12] = self.gas_demand[initialized_env_step][2]
        initial_states[self.operator_name[1]][13] = self.gas_demand[initialized_env_step][1]

        ### HSO initialized state
        np.random.seed(self.seed)
        # initial_states[self.operator_name[2]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        initial_states[self.operator_name[2]][0] = 0
        initial_states[self.operator_name[2]][1] = 0
        initial_states[self.operator_name[2]][2] = 0
        initial_states[self.operator_name[2]][3] = self.heat_demand[initialized_episode_step, initialized_env_step].sum()
        initial_states[self.operator_name[2]][4] = 0
        
        self.states = initial_states

        state = self.flatten_states(self.states)

        done = False

        return state, self.IM(copy.deepcopy(state)), done
    
    
    def reset_SOC(self, epoch):
        
        iteration = epoch
        iteration_hso = iteration

        initialized_env_step = 0

        
        ### PSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[0]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[0]][0] = 0
        self.states[self.operator_name[0]][1] = 0
        self.states[self.operator_name[0]][2] = 0
        self.states[self.operator_name[0]][3] = self.electricity_price[initialized_env_step] 
        self.states[self.operator_name[0]][4] = self.electricity_price[initialized_env_step] / 2
        self.states[self.operator_name[0]][5] = self.pv[iteration][initialized_env_step] * 0.9
        self.states[self.operator_name[0]][6] = self.pv[iteration][initialized_env_step] * 1.0
        self.states[self.operator_name[0]][7] = self.pv[iteration][initialized_env_step] * 1.1
        self.states[self.operator_name[0]][8] = self.pv[iteration][initialized_env_step] * 1.2
        self.states[self.operator_name[0]][9] = self.pv[iteration][initialized_env_step] * 1.3
        self.states[self.operator_name[0]][10] = self.wind[iteration][initialized_env_step] * 1.0
        self.states[self.operator_name[0]][11] = self.wind[iteration][initialized_env_step] * 1.1
        self.states[self.operator_name[0]][12] = self.wind[iteration][initialized_env_step] * 1.2
        self.states[self.operator_name[0]][13] = self.energy_demand[iteration][initialized_env_step].sum()
        self.states[self.operator_name[0]][14] = self.energy_demand_q[iteration][initialized_env_step].sum()


        ### GSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[1]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[1]][0] = 0
        self.states[self.operator_name[1]][1] = 0
        self.states[self.operator_name[1]][2] = 0
        # self.states[self.operator_name[1]][1] = self.gas_demand[initialized_env_step].sum()
        self.states[self.operator_name[1]][3] = self.gas_demand[initialized_env_step][0]
        self.states[self.operator_name[1]][4] = self.gas_demand[initialized_env_step][0]
        self.states[self.operator_name[1]][5] = self.gas_demand[initialized_env_step][1]
        self.states[self.operator_name[1]][6] = self.gas_demand[initialized_env_step][3]
        self.states[self.operator_name[1]][7] = self.gas_demand[initialized_env_step][2]
        self.states[self.operator_name[1]][8] = self.gas_demand[initialized_env_step][3]
        self.states[self.operator_name[1]][9] = self.gas_demand[initialized_env_step][2]
        self.states[self.operator_name[1]][10] = self.gas_demand[initialized_env_step][2]
        self.states[self.operator_name[1]][11] = self.gas_demand[initialized_env_step][3]
        self.states[self.operator_name[1]][12] = self.gas_demand[initialized_env_step][2]
        self.states[self.operator_name[1]][13] = self.gas_demand[initialized_env_step][1]


        ### HSO reset SOC and transit to next state
        np.random.seed(self.seed)
        # self.states[self.operator_name[2]][0] = float('%.2f' % np.random.normal(loc=0.25, scale=0.01, size=None))
        self.states[self.operator_name[2]][0] = 0
        self.states[self.operator_name[2]][1] = 0
        self.states[self.operator_name[2]][2] = 0
        self.states[self.operator_name[2]][3] = self.heat_demand[iteration_hso][initialized_env_step].sum()
        self.states[self.operator_name[2]][4] = 0

        state = self.flatten_states(self.states)
        done = False

        return state, self.IM(copy.deepcopy(state)), done


    def flatten_states(self, ori_states):
        for count, name in zip(range(len(self.operator_name)), self.operator_name):
            state = ori_states[name]
            if count == 0:
                flatten_states = state
            else:
                flatten_states = np.concatenate((flatten_states, state), axis = 0)
        return flatten_states

    def get_obs_dim(self):
        return int(self.obs_dim)

    def get_action_dim(self):
        return int(self.action_dim)
    
    def get_device_config(self):
        device_config = {}
        device_config['pso'] = np.array([self.dg_power_max,  self.gt_power_max, self.dg_q_power_max,  self.gt_q_power_max, self.es_power_max, self.es_capacity, self.es_efficiency])
        device_config['gso'] = np.array([self.gw1_power_max, self.gw2_power_max, self.gs_power_max, self.gs_capacity, self.gs_efficiency])
        device_config['hso'] = np.array([self.ehp_power_max, self.gb_power_max, self.ts_power_max, self.ts_capacity, self.ts_efficiency])

        return device_config
    
    def get_load(self, epoch, env_step):
        iteration = epoch
        iteration_hso = iteration
            
        load = {}
        load['pso'] = self.energy_demand[iteration][env_step]
        load['pso_q'] = self.energy_demand_q[iteration][env_step]
        load['gso'] = self.gas_demand[env_step]
        load['hso'] = self.heat_demand[iteration_hso][env_step]

        return load
    
    # def get_load_scenario(self, epoch, env_step):
    #     iteration = epoch % 334
    #     load = {}
    #     load['pso'] = self.energy_demand[iteration][env_step]
    #     load['gso'] = self.gas_demand[env_step]
    #     load['hso'] = self.heat_demand[iteration][env_step]

    #     return load
    

    def generate_haar_random_matrix(self, n, seed):
        np.random.seed(seed)
        random_matrix = np.random.randn(n, n)
        Q, R = np.linalg.qr(random_matrix)
        sign_diag = np.sign(np.diag(R))
        Q_haar = Q * sign_diag
        return Q_haar

    def generate_invertible_matrix(self, size, seed):
        while True:
            np.random.seed(seed)
            matrix = np.random.rand(size, size)
            if np.linalg.det(matrix) != 0:
                return matrix

    def rotate(self, state, seed, loc):
        n = len(loc)
        key = self.generate_haar_random_matrix(n, seed)
        s = np.zeros(n)
        for i, j in zip(loc,range(n)):
            s[j] = state[i]  
        s = np.dot(key, s)
        for i, j in zip(loc,range(n)):
            state[i] = s[j]
        return state

    def IM(self, input_states):
        # privacy module
        states = copy.deepcopy(input_states)

        # pso privacy module
        seed = 0
        loc = [0,1,2]
        states = self.rotate(states, seed, loc)

        seed += 1
        loc = [5,6,7,8,9,10,11,12,13,14]
        states = self.rotate(states, seed, loc)

        # seed += 1
        # lower_bound = [0,0,0,0,0,0,0]
        # upper_bound = [5,5,5,5,5,5,5]
        # np.random.seed(seed)
        # random_vector = np.random.uniform(lower_bound, upper_bound, self.obs_num[0])
        # states[:self.obs_num[0]] += random_vector

        # gso privacy module
        seed += 1
        loc = [18,19,20,21,22,23,24,25,26,27,28]
        states = self.rotate(states, seed, loc)

        seed += 1
        loc = [15,16,17]
        states = self.rotate(states, seed, loc)

        # hso privacy module
        seed += 1
        loc = [29,30,31,32]
        states = self.rotate(states, seed, loc)

        return states


