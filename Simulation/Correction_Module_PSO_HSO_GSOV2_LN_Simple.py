import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import time


class Correction_Module:                                  # correction module 的应用顺序应该是热——>电——>气（由于能量转换机组使网络耦合的缘故，一个网络的负荷是另一个网络的供给，因此需按顺序进行纠正）
    def __init__(self, state, action, device_config, load, test=False):
        # PSO
        self.energy_demand = np.array([0, load['pso'][0], load['pso'][1], load['pso'][2], 0, load['pso'][0],
                                       load['pso'][1], 0, load['pso'][2], load['pso'][0], 0, load['pso'][1],
                                       0, load['pso'][2], 0, load['pso'][0], load['pso'][1], load['pso'][2],
                                       load['pso'][0], 0, 0, load['pso'][1], 0, 0,
                                       load['pso'][2], 0, load['pso'][0], 0, load['pso'][1], load['pso'][2],
                                       load['pso'][0], load['pso'][1], load['pso'][2]])     # A varying input from environment
        self.energy_demand_q = np.array([0, load['pso_q'][0], load['pso_q'][1], load['pso_q'][2], 0, load['pso_q'][0],
                                       load['pso_q'][1], 0, load['pso_q'][2], load['pso_q'][0], 0, load['pso_q'][1],
                                       0, load['pso_q'][2], 0, load['pso_q'][0], load['pso_q'][1], load['pso_q'][2],
                                       load['pso_q'][0], 0, 0, load['pso_q'][1], 0, 0,
                                       load['pso_q'][2], 0, load['pso_q'][0], 0, load['pso_q'][1], load['pso_q'][2],
                                       load['pso_q'][0], load['pso_q'][1], load['pso_q'][2]])     # A varying input from environment
        self.energy_demand /= 2     #!!!
        self.energy_demand_q /= 2   #!!!
        self.pv_output = [state[5], state[6], state[7], state[8], state[9]]
        self.wt_output = [state[10], state[11], state[12]]
        self.SOC_energy = [state[0], state[1], state[2]]
        
        # GSO 
        self.gas_demand = np.array([load['gso'][0], load['gso'][0], 0, 0, 0,
                                    load['gso'][1], 0, 0, 0, 0,
                                    load['gso'][1], load['gso'][2], load['gso'][3], load['gso'][2], load['gso'][3],
                                    0, load['gso'][2], load['gso'][3], load['gso'][2], 0,])    # gas load of each node       
        self.SOC_gas = [state[15], state[16], state[17]]

        # self.gas_demand /= 2    #!!!

        # HSO
        self.heat_demand = np.array([0, 0, 0, 0, load['hso'][0], load['hso'][1], load['hso'][2], load['hso'][3]])               
        self.SOC_heat = [state[29], state[30], state[31]]

        # self.heat_demand /= 2       #!!!

        # configuration of devices
        self.action = action
        self.device_config = device_config
        self.test = test


    def generateCmatrix(self, n_bus, br_p):                       # 生成联系矩阵，1表示两个节点有链接，0表示无链接
        Connection = np.zeros((n_bus, n_bus)) 
        for branch in br_p:
            Connection[branch[0].astype(int) - 1][branch[1].astype(int) - 1] = 1
            Connection[branch[1].astype(int) - 1][branch[0].astype(int) - 1] = 1
        return Connection

    def generateRmatrix(self, n_bus, br_p):                       # 生成电阻R矩阵
        R = np.zeros((n_bus, n_bus)) 
        for branch in br_p:
            R[branch[0].astype(int) - 1][branch[1].astype(int) - 1] = branch[2]
            R[branch[1].astype(int) - 1][branch[0].astype(int) - 1] = branch[2]
        return R

    def generateXmatrix(self, n_bus, br_p):                       # 生成电抗X矩阵
        X = np.zeros((n_bus, n_bus))
        for branch in br_p:
            X[branch[0].astype(int) - 1][branch[1].astype(int) - 1] = branch[3]/3
            X[branch[1].astype(int) - 1][branch[0].astype(int) - 1] = branch[3]/3
        return X

    # The two functions below are for hso_correction module
    def generateMmatrix(self, n_node, br_sr, cp):              # branch with a shape of 3, 0: from node; 1: to node; 2: flow rate; 3:管道每米传输阻抗; 4: length
        M = np.zeros((n_node, n_node, 2))
        for branch in br_sr:
            M[branch[0].astype(int) - 1][branch[1].astype(int) - 1][0] = branch[2]
            M[branch[0].astype(int) - 1][branch[1].astype(int) - 1][1] = math.exp(-branch[3] * branch[4] / (cp * branch[2]))
        return M

    def generateTypeArrays(self, node_list):
        node_type1 = np.empty(shape=[0, 2])
        node_type2 = np.empty(shape=[0, 2])
        node_type3 = np.empty(shape=[0, 2])
        for node in node_list:
            if node[1] == 1:
                node_type1 = np.append(node_type1, [node], axis=0)
            elif node[1] == 2:
                node_type2 = np.append(node_type2, [node], axis=0)
            else:
                node_type3 = np.append(node_type3, [node], axis=0)
        
        return node_type1, node_type2, node_type3


    def pso_correction(self, load_ehp):
        ## inputs
        el = self.energy_demand                              # A varying input from environment
        el_q = self.energy_demand_q                              # A varying input from environment
        wt_output = self.wt_output                           # A varying input from environment
        pv_output =  self.pv_output                          # A varying input from environment
        SOC = self.SOC_energy                                   #SOC of the energy storage

        el[32] += load_ehp[0]
        el[17] += load_ehp[1]
        el[18] += load_ehp[2]

        # el_q[32] += load_ehp[0] * np.tan(np.arccos(0.95))
        # el_q[17] += load_ehp[1] * np.tan(np.arccos(0.95))
        # el_q[18] += load_ehp[2] * np.tan(np.arccos(0.95))

        #Action formulation (因为policy决策只考虑机组的输出（输出值在0-1间），在确定机组输出及负荷后，与UPG的交互是确定的。但在考虑网络约束时，所有的输出都需要优化。因此action需要重新构建，气网同理)
        action_dg1 = self.action[0] * self.device_config['pso'][0]                #得从Env中取设别参数，在Env中写入一个取参数的function
        action_dg2 = self.action[1] * self.device_config['pso'][0]
        action_gt1 = self.action[2] * self.device_config['pso'][1]
        action_gt2 = self.action[3] * self.device_config['pso'][1]
        action_dg_q1 = self.action[4] * self.device_config['pso'][2]
        action_dg_q2 = self.action[5] * self.device_config['pso'][2]
        action_gt_q1 = self.action[6] * self.device_config['pso'][3]
        action_gt_q2 = self.action[7] * self.device_config['pso'][3]
        action_es1 = - self.action[8] * self.device_config['pso'][4]         # correction_module的充放电设置和Env中是相反的，因此加上“-”号。表示正值为放电，负值为充电。
        action_es2 = - self.action[9] * self.device_config['pso'][4]
        action_es3 = - self.action[10] * self.device_config['pso'][4]
        action_upg = el.sum() - np.sum(wt_output) - np.sum(pv_output) - action_dg1 - action_dg2 - action_gt1 - action_gt2 - action_es1 - action_es2 - action_es3
        action_upg_q = el_q.sum() - action_dg_q1 - action_dg_q2 - action_gt_q1 - action_gt_q2              
        action_p = np.array([action_upg, action_upg_q, action_dg1, action_dg2, action_gt1, action_gt2, action_dg_q1, action_dg_q2, action_gt_q1, action_gt_q2, action_es1, action_es2, action_es3])          # The action needed to be tested and corrected, [UPG_p, UPG_q, DG_p, GT_p, DG_q, GT_q, STORAGE]

        ## hyperparameter
        n_bus = 33
        balance_node = 1
        p_factor = 0.95
        p_tran_q = np.tan(np.arccos(p_factor))
        BASE_P = 100
        BASE_Q = BASE_P * p_tran_q
        V_REF = 1
        capacity_storage = [self.device_config['pso'][5], self.device_config['pso'][5], self.device_config['pso'][5]]
        maxp_storage = [self.device_config['pso'][4], self.device_config['pso'][4], self.device_config['pso'][4]]
        efficiency = self.device_config['pso'][6]
        ubp_storage1 = min(capacity_storage[0] * SOC[0] *efficiency, maxp_storage[0])
        lbp_storage1 = max((SOC[0] - 1) * capacity_storage[0] /efficiency, -maxp_storage[0])
        ubp_storage2 = min(capacity_storage[1] * SOC[1] *efficiency, maxp_storage[1])
        lbp_storage2 = max((SOC[1] - 1) * capacity_storage[1] /efficiency, -maxp_storage[1])
        ubp_storage3 = min(capacity_storage[2] * SOC[2] *efficiency, maxp_storage[2])
        lbp_storage3 = max((SOC[2] - 1) * capacity_storage[2] /efficiency, -maxp_storage[2])
        
        g_list = np.array([[-999, 999],         #generator operation bound: upg
                        [0, self.device_config['pso'][0]],          #DG_p
                        [0, self.device_config['pso'][1]],          #GT_p
                        [0, self.device_config['pso'][2]],          #DG_q
                        [0, self.device_config['pso'][3]],          #GT_q
                        [lbp_storage1, ubp_storage1],               #Storage1
                        [lbp_storage2, ubp_storage2],               #Storage2
                        [lbp_storage3, ubp_storage3]])              #Storage3

        br_p = np.array([[1, 2, 0.017, 0.017],         #line reactance matrix
                        [2, 3, 0.0258, 0.0258],
                        [3, 4, 0.0197, 0.0197],
                        [4, 5, 0.018, 0.018],
                        [5, 6, 0.027, 0.027],
                        [6, 7, 0.027, 0.027],
                        [7, 8, 0.014, 0.014],
                        [8, 9, 0.017, 0.017],
                        [9, 10, 0.0258, 0.0258],
                        [10, 11, 0.0197, 0.0197],
                        [11, 12, 0.018, 0.018],
                        [12, 13, 0.027, 0.027],
                        [13, 14, 0.017, 0.017],
                        [14, 15, 0.014, 0.014],
                        [15, 16, 0.017, 0.017],
                        [16, 17, 0.0258, 0.0258],
                        [17, 18, 0.0197, 0.0197],
                        [19, 20, 0.018, 0.018],
                        [20, 21, 0.027, 0.027],
                        [21, 22, 0.027, 0.027],
                        [23, 24, 0.014, 0.014],
                        [24, 25, 0.017, 0.017],
                        [26, 27, 0.0258, 0.0258],
                        [27, 28, 0.0197, 0.0197],
                        [28, 29, 0.018, 0.018],
                        [29, 30, 0.017, 0.017],
                        [30, 31, 0.027, 0.027],
                        [31, 32, 0.014, 0.014],
                        [32, 33, 0.017, 0.017],
                        [3, 23, 0.018, 0.018],
                        [6, 26, 0.017, 0.017],
                        [2, 19, 0.017, 0.017]])

        br_flow_constr = np.array([[1, 2, 600],         #line flow constraints
                                    [2, 3, 500],
                                    [3, 4, 500],
                                    [4, 5, 500],
                                    [5, 6, 500],
                                    [6, 7, 500],
                                    [7, 8, 500],
                                    [8, 9, 500],
                                    [9, 10, 500],
                                    [10, 11, 500],
                                    [11, 12, 500],
                                    [12, 13, 500],
                                    [13, 14, 500],
                                    [14, 15, 500],
                                    [15, 16, 500],
                                    [16, 17, 500],
                                    [17, 18, 500],
                                    [19, 20, 500],
                                    [20, 21, 500],
                                    [21, 22, 500],
                                    [23, 24, 500],
                                    [24, 25, 500],
                                    [26, 27, 500],
                                    [27, 28, 500],
                                    [28, 29, 500],
                                    [29, 30, 500],
                                    [30, 31, 500],
                                    [31, 32, 500],
                                    [32, 33, 500],
                                    [3, 23, 500],
                                    [6, 26, 500],
                                    [2, 19, 500]])

        max_line_flow = np.zeros((n_bus,n_bus))                 #line flow constraint matrix
        for branch in br_flow_constr:
            max_line_flow[branch[0]-1, branch[1]-1] = branch[2]
            max_line_flow[branch[1]-1, branch[0]-1] = branch[2]

        #Optimizer
        m = gp.Model('PowerNetworkCorrection')
        m.setParam('OutputFlag', 0)

        p_upg = m.addVar(lb=g_list[0][0], ub=g_list[0][1], name='power output of UPG')
        p_g1 = m.addVar(lb=g_list[1][0], ub=g_list[1][1], name='power output of DG1')
        p_g2 = m.addVar(lb=g_list[1][0], ub=g_list[1][1], name='power output of DG2')
        p_g3 = m.addVar(lb=g_list[2][0], ub=g_list[2][1], name='power output of GT1')  
        p_g4 = m.addVar(lb=g_list[2][0], ub=g_list[2][1], name='power output of GT2')
        q_upg = m.addVar(lb=g_list[0][0], ub=g_list[0][1], name='reactive power output of UPG')
        q_g1 = m.addVar(lb=g_list[3][0], ub=g_list[3][1], name='reactive power output of DG1')
        q_g2 = m.addVar(lb=g_list[3][0], ub=g_list[3][1], name='reactive power output of DG2')
        q_g3 = m.addVar(lb=g_list[4][0], ub=g_list[4][1], name='reactive power output of GT1')  
        q_g4 = m.addVar(lb=g_list[4][0], ub=g_list[4][1], name='reactive power output of GT2')
        p_storage1 = m.addVar(lb=g_list[5][0], ub=g_list[5][1], name='power output of Storage1')  
        p_storage2 = m.addVar(lb=g_list[6][0], ub=g_list[6][1], name='power output of Storage2')  
        p_storage3 = m.addVar(lb=g_list[7][0], ub=g_list[7][1], name='power output of Storage3')    

        v_node = {}
        for i in range(n_bus):
            v_node[i] = m.addVar(lb=0.90, ub = 1.10, name='voltage angle from bus 1 to bus 6')
        v_node[balance_node] = V_REF
        P_ex = m.addVars(n_bus,n_bus, lb=-999, ub=999, name='power exchange from bus 1 to bus 6')
        Q_ex = m.addVars(n_bus,n_bus, lb=-999, ub=999, name='reactive power exchange from bus 1 to bus 6')

        # 重组数据
        p_g_set = np.array([0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    p_g3, 0, 0, 0, 0, 0,
                    0, p_g4, 0, 0, 0, p_g1,
                    0, 0, 0, 0, 0, 0,
                    p_g2, 0, 0])
        q_g_set = np.array([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            q_g3, 0, 0, 0, 0, 0,
                            0, q_g4, 0, 0, 0, q_g1,
                            0, 0, 0, 0, 0, 0,
                            q_g2, 0, 0])
        p_upg_set = np.array([p_upg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        q_upg_set = np.array([q_upg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        p_wt_set = np.array([0, 0, 0, 0, 0, 0,
                            0, wt_output[1], 0, 0, 0, 0,
                            0, 0, wt_output[2], 0, 0, 0, 
                            0, 0, 0, 0, 0, 0,
                            0, wt_output[1], 0, 0, 0, 0,
                            0, 0, 0])
        p_pv_set = np.array([0, 0, 0, 0, pv_output[2], 0,
                            0, 0, 0, 0, pv_output[3], 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, pv_output[4], 0, pv_output[0], 0, 
                            0, 0, 0, pv_output[1], 0, 0, 
                            0, 0, 0])
        p_storage_set = np.array([0, 0, 0, 0, 0, p_storage1,
                                0, 0, 0, 0, 0, 0, 
                                0, 0, 0, p_storage2, 0, 0,
                                0, 0, 0, 0, 0, 0, 
                                0, 0, 0, p_storage3, 0, 0, 
                                0, 0, 0])


        # 生成邻接矩阵
        R = self.generateRmatrix(n_bus, br_p) 
        X = self.generateXmatrix(n_bus, br_p)     
        C = self.generateCmatrix(n_bus, br_p)       

        for i in range(n_bus):
            m.addConstr((p_g_set[i] + p_upg_set[i] + p_wt_set[i] + p_pv_set[i] + p_storage_set[i])
                        == el[i] + gp.quicksum(P_ex[i,j]*C[i,j] for j in range(n_bus)))
            
            m.addConstr(q_g_set[i] + q_upg_set[i] == el_q[i] + gp.quicksum(Q_ex[i,j]*C[i,j] for j in range(n_bus)))
        
        # for i in range(n_bus):
        #     for j in range(n_bus):
        #         m.addConstr((v_node[j] - v_node[i]) * C[i,j] == - (R[i,j]*P_ex[i,j]/BASE_P + X[i,j]*Q_ex[i,j]/BASE_Q)/V_REF)

        for i in range(n_bus):
            for j in range(n_bus):
                m.addConstr(P_ex[i,j] == -P_ex[j,i])
                m.addConstr(Q_ex[i,j] == -Q_ex[j,i])

        for i in range(n_bus):
            for j in range(n_bus):
                m.addConstr( P_ex[i,j] <= max_line_flow[i,j] * C[i,j])
                m.addConstr( P_ex[i,j] >= -max_line_flow[i,j] * C[i,j])
                m.addConstr( Q_ex[i,j] <= max_line_flow[i,j]*p_tran_q * C[i,j])
                m.addConstr( Q_ex[i,j] >= -max_line_flow[i,j]*p_tran_q * C[i,j])

        m.addConstr(q_g1 <= p_g1 * p_tran_q)
        m.addConstr(q_g2 <= p_g2 * p_tran_q)

        # 优化目标方程
        m.setObjective(gp.quicksum(([p_g1, p_g2, p_g3, p_g4, q_g1, q_g2, q_g3, q_g4, p_storage1, p_storage2, p_storage3][i-2] - action_p[i]) ** 2 for i in range(2,13)), GRB.MINIMIZE)   #目标方程不能使用numpy L2-norm
        m.setParam("MIPFocus", 1)
        m.setParam("TimeLimit", 10.0)
        m.setParam("Presolve", 0)
        m.setParam("Cuts", 0)
        m.setParam("MIPGap", 0.5)
        m.setParam("FeasibilityTol", 0.01)
        m.setParam("OptimalityTol", 0.01)
        m.optimize()

        epsilon = 0.0001

        # if m.objVal <= epsilon:
        #     action_safe = action_p
        # else:
        #     generator_output = [p_upg.X, p_g1.X, p_storage.X, p_g2.X]
        #     action_safe = np.array(generator_output)

        OV = m.objVal
        generator_output = [p_upg.X, q_upg.X, p_g1.X, p_g2.X, p_g3.X, p_g4.X, q_g1.X, q_g2.X, q_g3.X, q_g4.X, p_storage1.X, p_storage2.X, p_storage3.X]
        action_safe = np.array(generator_output)
        
        if self.test:
            pso_flow_p = []
            pso_flow_q = []
            for branch in br_p:
                pso_flow_p.append(abs(P_ex[branch[0]-1, branch[1]-1].X))
                pso_flow_q.append(abs(Q_ex[branch[1]-1, branch[0]-1].X))
            
            Voltage = []
            for i in range(n_bus):
                if i != balance_node:
                    Voltage.append(v_node[i].X)
                else:
                    Voltage.append(1)
        else:
            pso_flow_p = None
            pso_flow_q = None
            Voltage = None
        
        return action_safe, OV, pso_flow_p, pso_flow_q, Voltage
    
    def gso_correction(self, load_gt, load_gb):
        ## inputs
        gl = self.gas_demand    # gas load
        SOC = self.SOC_gas
        gl[6] += load_gt[0]
        gl[3] += load_gt[1]
        gl[5] += load_gb[0]
        gl[17] += load_gb[1]
        gl[0] += load_gb[2]

        #Action formulation
        action_gw1 = self.action[11] * self.device_config['gso'][0]
        action_gw2 = self.action[12] * self.device_config['gso'][0]
        action_gw3 = self.action[13] * self.device_config['gso'][1]
        action_gw4 = self.action[14] * self.device_config['gso'][1]
        action_gs1 = - self.action[15] * self.device_config['gso'][2]
        action_gs2 = - self.action[16] * self.device_config['gso'][2]
        action_gs3 = - self.action[17] * self.device_config['gso'][2]
        action_ugg = gl.sum() - action_gw1 - action_gw2 - action_gw3 - action_gw4 - action_gs1 - action_gs2 - action_gs3

        action_p = np.array([action_ugg, action_gw1, action_gw2, action_gw3, action_gw4, action_gs1, action_gs2, action_gs3])          # The action needed to be tested and corrected, [UGG, GW1, GW2, STORAGE]

        ## hyperparameter
        n_node_gso = 20
        balance_node = 3
        n_pipeline = 20
        lambda_l = 4                                                #取值范围一般为1.5-4
        capacity_storage = [self.device_config['gso'][3], self.device_config['gso'][3], self.device_config['gso'][3]]
        maxp_storage = [self.device_config['gso'][2], self.device_config['gso'][2], self.device_config['gso'][2]]
        efficiency = self.device_config['gso'][4]
        ubp_storage1 = min(capacity_storage[0] * SOC[0] *efficiency, maxp_storage[0])
        lbp_storage1 = max((SOC[0] - 1) * capacity_storage[0] /efficiency, -maxp_storage[0])
        ubp_storage2 = min(capacity_storage[1] * SOC[1] *efficiency, maxp_storage[1])
        lbp_storage2 = max((SOC[1] - 1) * capacity_storage[1] /efficiency, -maxp_storage[1])
        ubp_storage3 = min(capacity_storage[2] * SOC[2] *efficiency, maxp_storage[2])
        lbp_storage3 = max((SOC[2] - 1) * capacity_storage[2] /efficiency, -maxp_storage[2])

        g_list = np.array([[0, 999],         #generator operation bound: ugg
                        [0, self.device_config['gso'][0]],          #GW1
                        [0, self.device_config['gso'][0]],          #GW2
                        [0, self.device_config['gso'][1]],          #GW3
                        [0, self.device_config['gso'][1]],          #GW4
                        [lbp_storage1, ubp_storage1],
                        [lbp_storage2, ubp_storage2],
                        [lbp_storage3, ubp_storage3]])          #Storage

        pipeline_flow_constr = np.array([[1, 2, 300],         #line flow constraints
                                 [2, 3, 300],
                                 [3, 4, 300],
                                 [4, 7, 300],
                                 [7, 6, 300],
                                 [6, 5, 300],
                                 [16, 15, 300],
                                 [15, 8, 300],
                                 [8, 14, 300],
                                 [14, 13, 300],
                                 [13, 11, 300],
                                 [11, 10, 300],
                                 [10, 9, 300],
                                 [13, 12, 300],
                                 [12, 17, 300],
                                 [17, 18, 300],
                                 [18, 19, 300],
                                 [19, 20, 300],
                                 [4, 8, 300],
                                 [7, 10, 300]])

        pipeline_para = np.array([[1, 2, 1, 6.2640],         ##line parameter: from node, to node, node type(1-active;0-inactive), C_mn
                                [2, 3, 1, 6.2640],
                                [3, 4, 1, 6.2640],
                                [4, 7, 1, 9.3960],
                                [7, 6, 1, 6.2640],
                                [6, 5, 1, 6.2640],
                                [16, 15, 1, 6.2640],
                                [15, 8, 1, 6.2640],
                                [8, 14, 1, 6.2640],
                                [14, 13, 1, 6.2640],
                                [13, 11, 1, 3.7584],
                                [11, 10, 1, 6.2640],
                                [10, 9, 1, 6.2640],
                                [13, 12, 1, 3.7584],
                                [12, 17, 1, 6.2640],
                                [17, 18, 1, 6.2640],
                                [18, 19, 1, 6.2640],
                                [19, 20, 1, 6.2640],
                                [4, 8, 1, 3.7584],
                                [7, 10, 1, 9.3960]])

        max_pipeline_flow = np.zeros((n_node_gso,n_node_gso))                 #pipeline flow constraint matrix
        for pipeline in pipeline_flow_constr:
            max_pipeline_flow[pipeline[0]-1, pipeline[1]-1] = pipeline[2]
            max_pipeline_flow[pipeline[1]-1, pipeline[0]-1] = pipeline[2]

        #Optimizer
        m = gp.Model('GasNetworkCorrection')
        m.setParam('OutputFlag', 0)

        f_ugg = m.addVar(lb=g_list[0][0], ub=g_list[0][1], name='power output of UGG')  
        f_gw1 = m.addVar(lb=g_list[1][0], ub=g_list[1][1], name='power output of GAS WELL 1')
        f_gw2 = m.addVar(lb=g_list[2][0], ub=g_list[2][1], name='power output of GAS WELL 2')
        f_gw3 = m.addVar(lb=g_list[3][0], ub=g_list[3][1], name='power output of GAS WELL 3')
        f_gw4 = m.addVar(lb=g_list[4][0], ub=g_list[4][1], name='power output of GAS WELL 4')
        f_storage1 = m.addVar(lb=g_list[5][0], ub=g_list[5][1], name='power output of Storage1')
        f_storage2 = m.addVar(lb=g_list[6][0], ub=g_list[6][1], name='power output of Storage2')   
        f_storage3 = m.addVar(lb=g_list[7][0], ub=g_list[7][1], name='power output of Storage3')       

        f_pipeline = m.addVars(n_pipeline, lb=0, ub=pipeline_flow_constr[5,2], name='gas flow of pipeline')  # f_pipeline 1-7 的建模
        d_pipeline = m.addVars(n_pipeline, vtype=GRB.BINARY)

        p_node = {}
        p_node_sq = {}
        for i in range(n_node_gso):
            p_node[i] = m.addVar(lb=20, ub=200, name='gas pressure of node')
            p_node_sq[i] = m.addVar(lb=400, ub=40000, name='gas pressure of node')
        p_node[balance_node] = 70
        p_node_sq[balance_node] = 70*70
        
        # 重组数据
        f_gw_set = np.array([0, 0, 0, 0, f_gw1, 0, 0, 0, f_gw4, 0, 0, 0, 0, 0, 0, f_gw2, 0, 0, 0, f_gw3])
        f_ugg_set = np.array([f_ugg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        f_storage_set = np.array([0, 0, f_storage1, 0, 0, 0, 0, f_storage2, 0, f_storage3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        f_pipeline_set = np.array([-f_pipeline[0]*(2*d_pipeline[0]-1),
                                    f_pipeline[0]*(2*d_pipeline[0]-1) - f_pipeline[1]*(2*d_pipeline[1]-1),
                                    f_pipeline[1]*(2*d_pipeline[1]-1) - f_pipeline[2]*(2*d_pipeline[2]-1),
                                    f_pipeline[2]*(2*d_pipeline[2]-1) - f_pipeline[3]*(2*d_pipeline[3]-1) - f_pipeline[18]*(2*d_pipeline[18]-1),
                                    f_pipeline[5]*(2*d_pipeline[5]-1),
                                    f_pipeline[4]*(2*d_pipeline[4]-1) - f_pipeline[5]*(2*d_pipeline[5]-1),
                                    f_pipeline[3]*(2*d_pipeline[3]-1) - f_pipeline[4]*(2*d_pipeline[4]-1) - f_pipeline[19]*(2*d_pipeline[19]-1),
                                    f_pipeline[7]*(2*d_pipeline[7]-1) + f_pipeline[18]*(2*d_pipeline[18]-1) - f_pipeline[8]*(2*d_pipeline[8]-1),
                                    f_pipeline[12]*(2*d_pipeline[12]-1),
                                    f_pipeline[19]*(2*d_pipeline[19]-1) + f_pipeline[11]*(2*d_pipeline[11]-1) - f_pipeline[12]*(2*d_pipeline[12]-1),
                                    f_pipeline[10]*(2*d_pipeline[10]-1) - f_pipeline[11]*(2*d_pipeline[11]-1),
                                    f_pipeline[13]*(2*d_pipeline[13]-1) - f_pipeline[14]*(2*d_pipeline[14]-1),
                                    f_pipeline[9]*(2*d_pipeline[9]-1) - f_pipeline[10]*(2*d_pipeline[10]-1) - f_pipeline[13]*(2*d_pipeline[13]-1),
                                    f_pipeline[8]*(2*d_pipeline[8]-1) - f_pipeline[9]*(2*d_pipeline[9]-1),
                                    f_pipeline[6]*(2*d_pipeline[6]-1) - f_pipeline[7]*(2*d_pipeline[7]-1),
                                    -f_pipeline[6]*(2*d_pipeline[6]-1),
                                    f_pipeline[14]*(2*d_pipeline[14]-1) - f_pipeline[15]*(2*d_pipeline[15]-1),
                                    f_pipeline[15]*(2*d_pipeline[15]-1) - f_pipeline[16]*(2*d_pipeline[16]-1),
                                    f_pipeline[16]*(2*d_pipeline[16]-1) - f_pipeline[17]*(2*d_pipeline[17]-1),
                                    f_pipeline[17]*(2*d_pipeline[17]-1)
                                    ])

        # 网络约束
        for i in range(n_pipeline):
            # if pipeline_para[i,2] == 0:
            #     fn = pipeline_para[i,0] - 1
            #     tn = pipeline_para[i,1] - 1
            #     u = gp.QuadExpr(2*d_pipeline[i]*p_node_sq[fn])
            #     v = gp.QuadExpr(2*d_pipeline[i]*p_node_sq[tn])
            #     w = gp.QuadExpr(f_pipeline[i]*f_pipeline[i])
            #     m.addConstr( w - pipeline_para[i,3]*(u - p_node_sq[fn] - v + p_node_sq[tn]) == 0 )                        # relationship constraints between gas flow and nodal gas pressure in each inactive gas pipelineline 
            # else:
            fn = pipeline_para[i,0] - 1
            tn = pipeline_para[i,1] - 1
            m.addConstr( d_pipeline[i]*p_node[fn] <= d_pipeline[i]*p_node[tn])
            m.addConstr( d_pipeline[i]*p_node[tn] <= d_pipeline[i]*lambda_l * p_node[fn])
            m.addConstr( (1-d_pipeline[i])*p_node[tn] <= (1-d_pipeline[i])*p_node[fn])
            m.addConstr( (1-d_pipeline[i])*p_node[fn] <= (1-d_pipeline[i])*lambda_l * p_node[tn])
        
        # for i in range(n_node_gso):
        #     m.addConstr(p_node_sq[i] == p_node[i] * p_node[i])
        
        m.addConstrs(
            f_ugg_set[i] + f_gw_set[i] + f_storage_set[i] + f_pipeline_set[i] == gl[i] for i in range(n_node_gso)
            )                                     # nodal gas balance constraints

        # 优化目标方程
        m.setObjective(gp.quicksum(([f_gw1, f_gw2, f_gw3, f_gw4, f_storage1, f_storage2, f_storage3][i-1] - action_p[i]) ** 2 for i in range(1,8)), GRB.MINIMIZE) 

        #Optimization Result
        m.params.NonConvex = 2
        m.setParam("MIPFocus", 1)
        m.setParam("TimeLimit", 5.0)
        m.setParam("Presolve", 0)
        m.setParam("Cuts", 0)
        m.setParam("MIPGap", 0.5)
        m.setParam("FeasibilityTol", 0.01)
        m.setParam("OptimalityTol", 0.01)
        
        # m.setParam("Threads", 0)
        m.optimize()

        # Action Correction
        epsilon = 0.0001

        # if m.objVal <= epsilon:
        #     action_safe = action_p
        # else:
        #     generator_output = [f_ugg.X, f_gw1.X, f_gw2.X, f_storage.X]
        #     action_safe = np.array(generator_output)
        
        OV = m.objVal
        generator_output = [f_ugg.X, f_gw1.X, f_gw2.X, f_gw3.X, f_gw4.X, f_storage1.X, f_storage2.X, f_storage3.X]
        action_safe = np.array(generator_output)
        
        if self.test:
            gso_flow = []
            for i in range(n_pipeline):
                gso_flow.append(f_pipeline[i].X)
            
            Pressure = []
            for i in range(n_node_gso):
                if i != balance_node:
                    Pressure.append(p_node[i].X)
                else:
                    Pressure.append(45)
        else:
            gso_flow = None
            Pressure = None
        
        return action_safe, OV, gso_flow, Pressure
    
    def hso_correction(self, action_hso_raw, SOC):
        
        ## inputs
        hl = self.heat_demand    # heat load / kJ
        
        t_amb = 10           # ！后续采取变化的日内室外温度！  

        # Action formulation
        # action_ehp = self.action[0] * self.device_config['hso'][0]
        # action_gb = self.action[1] * self.device_config['hso'][1]
        action_ehp = action_hso_raw[0]
        action_gb = action_hso_raw[1]
        action_ts = - action_hso_raw[2] * self.device_config['hso'][2]

        if action_gb == 0:
            action_gb = 0.0001
            action_ehp = 1 - action_gb
        if action_ehp == 0:
            action_ehp = 0.0001
            action_gb = 1 - action_ehp

        action_p = np.array([action_ehp, action_gb, action_ts])          # The action needed to be tested and corrected, [EHP, GB, STORAGE]

        ## hyperparameter
        n_node_hso = 8
        cp = 4200  # 水的比热容
        capacity_storage = self.device_config['hso'][3]
        maxp_storage = self.device_config['hso'][2]
        efficiency = self.device_config['hso'][4]
        ubp_storage = min(capacity_storage * SOC *efficiency, maxp_storage)
        lbp_storage = max((SOC - 1) * capacity_storage /efficiency, -maxp_storage)

        b_list = np.array([[0, self.device_config['hso'][0]],         #ehp
                        [0, self.device_config['hso'][1]],          #gb
                        [lbp_storage, ubp_storage]])          #Storage

        lb_ts = 60
        ub_ts = 90
        lb_tr = 30
        ub_tr = 70

        br_s = np.array([                              
                        [1, 2, 2.4, 0.03, 3500],
                        [2, 3, 1.8, 0.03, 1750],
                        [3, 4, 1.2, 0.03, 1750],
                        [2, 5, 0.6, 0.03, 1750],
                        [3, 6, 0.6, 0.03, 750],
                        [4, 7, 0.6, 0.03, 1750],
                        [4, 8, 0.6, 0.03, 750],
                            ])


        br_r = np.array([                              
                        [2, 1, 2.4, 0.03, 3500],
                        [3, 2, 1.8, 0.03, 1750],
                        [4, 3, 1.2, 0.03, 1750],
                        [5, 2, 0.6, 0.03, 1750],
                        [6, 3, 0.6, 0.03, 750],
                        [7, 4, 0.6, 0.03, 1750],
                        [8, 4, 0.6, 0.03, 750],
                            ])

        if action_p[2] > 0:
            storage_node_type = 1
        elif action_p[2] < 0:
            storage_node_type = 2
        else:
            storage_node_type = 3
            
        node_list = [[1, 1], 
                    [2, 3], 
                    [3, 3],
                    [4, 3], 
                    [5, 2], 
                    [6, 2], 
                    [7, 2], 
                    [8, 2]
                    ]

        ## Optimizer
        m = gp.Model('HeatNetworkCorrection')
        m.setParam('OutputFlag', 0)

        h_gb = m.addVar(lb=b_list[0][0], ub=b_list[0][1], name='heating output of gas boiler')
        h_ehp = m.addVar(lb=b_list[1][0], ub=b_list[1][1], name='heating output of electric heat pump')
        h_storage = m.addVar(lb=b_list[2][0], ub=b_list[2][1], name='heating output of thermal storage')

        ts = m.addVars(n_node_hso, vtype=GRB.CONTINUOUS, lb=lb_ts, ub=ub_ts, name='supply temperature from bus 1 to bus 8')
        tr = m.addVars(n_node_hso, vtype=GRB.CONTINUOUS, lb=lb_tr, ub=ub_tr, name='return temperature from bus 1 to bus 8')

        M_ts = self.generateMmatrix(n_node_hso, br_s, cp)
        M_tr = self.generateMmatrix(n_node_hso, br_r, cp)
        node_type1, node_type2, node_type3 = self.generateTypeArrays(node_list)

        # 重组数据
        h_ehp_set = np.array([h_ehp, 0, 0, 0, 0, 0, 0, 0])
        h_gb_set = np.array([h_gb, 0, 0, 0, 0, 0, 0, 0])
        h_storage_set = np.array([h_storage, 0, 0, 0, 0, 0, 0, 0])


        # energy conservation equation for the nodes of type 1
        m.addConstrs(cp / 1000 * sum(M_ts[node_type1[i][0].astype(int) - 1][j][0] for j in range(n_node_hso)) *                  # unit are: kJ, kW = kJ/s
                    (ts[node_type1[i][0].astype(int) - 1] - tr[node_type1[i][0].astype(int) - 1]) ==
                    h_ehp_set[0] + h_gb_set[0] + h_storage_set[0] for i in range(node_type1.shape[0]))

        # energy conservation equation for the nodes of type 2
        m.addConstrs(cp / 1000 * sum(M_ts[j][node_type2[i][0].astype(int) - 1][0] for j in range(n_node_hso)) * 
                    (ts[node_type2[i][0].astype(int) - 1] - tr[node_type2[i][0].astype(int) - 1]) ==
                    hl[node_type2[i][0].astype(int) - 1] for i in range(node_type2.shape[0]))

        # supply temperature mixed equation
        m.addConstrs(sum(M_ts[j][i][0] * ((ts[j] - t_amb) * M_ts[j][i][1] + t_amb) for j in range(n_node_hso)) ==
                    sum(M_ts[j][i][0] for j in range(n_node_hso)) * ts[i] for i in range(n_node_hso))

        # return temperature mixed equation
        m.addConstrs(sum(M_tr[j][i][0] * ((tr[j] - t_amb) * M_tr[j][i][1] + t_amb) for j in range(n_node_hso)) ==
                    sum(M_tr[j][i][0] for j in range(n_node_hso)) * tr[i] for i in range(n_node_hso))
        
        # 优化目标方程
        # m.setObjective(gp.quicksum(([h_ehp, h_gb, h_storage][i] - action_p[i]) ** 2 for i in range(3)), GRB.MINIMIZE)
        m.setObjective( (action_p[1]/action_p[0] * h_ehp - h_gb ) ** 2 + (h_storage - action_p[2]) **2, GRB.MINIMIZE)

        #Optimization Result
        m.optimize()

        # Action Correction
        epsilon = 0.0001
        # if m.objVal <= epsilon:
        #     action_safe = action_p
        # else:
        #     generator_output = [h_ehp.X, h_gb.X, h_storage.X]
        #     action_safe = np.array(generator_output)

        OV = m.objVal
        generator_output = [h_ehp.X, h_gb.X, h_storage.X]
        action_safe = np.array(generator_output)
        
        if self.test:
            temp_supply = []
            temp_return = []
            for i in range(n_node_hso):
                temp_supply.append(ts[i].X)
                temp_return.append(tr[i].X)
        else:
            temp_supply = None
            temp_return = None
        
        return action_safe, OV, temp_supply, temp_return
        
    def get_action(self):
        # action_safe=self.action

        ObjVal = []

        start_time = time.time()

        action_hso_raw1 = np.array([self.action[18], self.action[21], self.action[24]])
        action_hso_raw2 = np.array([self.action[19], self.action[22], self.action[25]])
        action_hso_raw3 = np.array([self.action[20], self.action[23], self.action[26]])
        action_hso1, ov1, temp_supply1, temp_return1 = self.hso_correction(action_hso_raw1, self.SOC_heat[0])
        action_hso2, ov2, temp_supply2, temp_return2 = self.hso_correction(action_hso_raw2, self.SOC_heat[1])
        action_hso3, ov3, temp_supply3, temp_return3 = self.hso_correction(action_hso_raw3, self.SOC_heat[2])
        temp_supply = [temp_supply1, temp_supply2, temp_supply3]
        temp_return = [temp_return1, temp_return2, temp_return3]
        ObjVal.append(ov1+ov2+ov3)

        # time1 = time.time()
        # execution_time1 = time1 - start_time
        # print(f"The execution time of hso_correction is {execution_time1} seconds.")

        load_ehp = np.array([action_hso1[0]/3, action_hso2[0]/3, action_hso3[0]/3])   #考虑能源转换效率
        action_pso, ov2, pso_flow_p, pso_flow_q, Voltage = self.pso_correction(load_ehp)
        ObjVal.append(ov2)

        # time2 = time.time()
        # execution_time2 = time2 - time1
        # print(f"The execution time of pso_correction is {execution_time2} seconds.")

        load_gt1 = action_pso[4] / 2                          #考虑能源转换效率
        load_gt2 = action_pso[5] / 2                          #考虑能源转换效率
        load_gb1 = action_hso1[1] / 5                          #考虑能源转换效率
        load_gb2 = action_hso2[1] / 5                          #考虑能源转换效率
        load_gb3 = action_hso3[1] / 5                          #考虑能源转换效率
        load_gt = np.array([load_gt1, load_gt2])
        load_gb = np.array([load_gb1, load_gb2, load_gb3])
        action_gso, ov3, gso_flow, Pressure = self.gso_correction(load_gt, load_gb)
        ObjVal.append(ov3)

        # time3 = time.time()
        # execution_time3 = time3 - time2
        # print(f"The execution time of gso_correction is {execution_time3} seconds.")

        # action conversion back into the primal form
        action_safe = np.zeros(27) 
        action_safe[0] = action_pso[2] / self.device_config['pso'][0]
        action_safe[1] = action_pso[3] / self.device_config['pso'][0]
        action_safe[2] = action_pso[4] / self.device_config['pso'][1]
        action_safe[3] = action_pso[5] / self.device_config['pso'][1]
        action_safe[4] = action_pso[6] / self.device_config['pso'][2]
        action_safe[5] = action_pso[7] / self.device_config['pso'][2]
        action_safe[6] = action_pso[8] / self.device_config['pso'][3]
        action_safe[7] = action_pso[9] / self.device_config['pso'][3]
        action_safe[8] = - (action_pso[10] / self.device_config['pso'][4])
        action_safe[9] = - (action_pso[11] / self.device_config['pso'][4])
        action_safe[10] = - (action_pso[12] / self.device_config['pso'][4])

        action_safe[11] = action_gso[1] / self.device_config['gso'][0]
        action_safe[12] = action_gso[2] / self.device_config['gso'][0]
        action_safe[13] = action_gso[3] / self.device_config['gso'][1]
        action_safe[14] = action_gso[4] / self.device_config['gso'][1]
        action_safe[15] = - (action_gso[5] / self.device_config['gso'][2])
        action_safe[16] = - (action_gso[6] / self.device_config['gso'][2])
        action_safe[17] = - (action_gso[7] / self.device_config['gso'][2])

        action_safe[18] = action_hso1[0] / self.device_config['hso'][0]
        action_safe[19] = action_hso2[0] / self.device_config['hso'][0]
        action_safe[20] = action_hso3[0] / self.device_config['hso'][0]
        action_safe[21] = action_hso1[1] / self.device_config['hso'][1]
        action_safe[22] = action_hso2[1] / self.device_config['hso'][1]
        action_safe[23] = action_hso3[1] / self.device_config['hso'][1]
        action_safe[24] = - (action_hso1[2] / self.device_config['hso'][2])
        action_safe[25] = - (action_hso2[2] / self.device_config['hso'][2])
        action_safe[26] = - (action_hso3[2] / self.device_config['hso'][2])

        main_grid_interaction = np.zeros(3)                    # [upg_p, upg_p, ugg]
        main_grid_interaction[0] = action_pso[0]
        main_grid_interaction[1] = action_pso[1]
        main_grid_interaction[2] = action_gso[0]

        
        # main_grid_interaction=np.zeros(3)
        # ObjVal=None 
        
        # return action_safe, main_grid_interaction, ObjVal

        if self.test:
            return action_safe, main_grid_interaction, ObjVal, pso_flow_p, pso_flow_q, Voltage, gso_flow, Pressure, temp_supply, temp_return
        else:
            return action_safe, main_grid_interaction, ObjVal






