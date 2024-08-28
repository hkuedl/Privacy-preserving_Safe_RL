import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math


class Correction_Module:                                  # correction module 的应用顺序应该是热——>电——>气（由于能量转换机组使网络耦合的缘故，一个网络的负荷是另一个网络的供给，因此需按顺序进行纠正）
    def __init__(self, state, action, device_config, load, test=False):
        # PSO
        self.energy_demand = np.array([0, 0, load['pso'][0], load['pso'][1], 0, load['pso'][2]]) 
        self.energy_demand_q = np.array([0, 0, load['pso_q'][0], load['pso_q'][1], 0, load['pso_q'][2]])          
        self.pv_output = state[3]
        self.wt_output = state[4]
        self.SOC_energy = state[0]
        
        # GSO
        self.gas_demand = np.array([load['gso'][0], 0, load['gso'][2], load['gso'][1], load['gso'][3], 0, 0])            
        self.SOC_gas = state[7]

        # HSO
        self.heat_demand = np.array([0, 0, 0, 0, load['hso'][0], load['hso'][1], load['hso'][2], load['hso'][3]])                  
        self.SOC_heat = state[12]

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
            X[branch[0].astype(int) - 1][branch[1].astype(int) - 1] = branch[3]
            X[branch[1].astype(int) - 1][branch[0].astype(int) - 1] = branch[3]
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

        el[5] += load_ehp


        #Action formulation (因为policy决策只考虑机组的输出（输出值在0-1间），在确定机组输出及负荷后，与UPG的交互是确定的。但在考虑网络约束时，所有的输出都需要优化。因此action需要重新构建，气网同理)
        action_dg = self.action[0] * self.device_config['pso'][0]                #得从Env中取设别参数，在Env中写入一个取参数的function
        action_gt = self.action[1] * self.device_config['pso'][1]
        action_dg_q = self.action[2] * self.device_config['pso'][2]
        action_gt_q = self.action[3] * self.device_config['pso'][3]
        action_es = - self.action[4] * self.device_config['pso'][4]         # correction_module的充放电设置和Env中是相反的，因此加上“-”号。表示正值为放电，负值为充电。
        action_upg = el.sum() - wt_output - pv_output - action_dg - action_gt - action_es
        action_upg_q = el_q.sum() - action_dg_q - action_gt_q              
        action_p = np.array([action_upg, action_upg_q, action_dg, action_gt, action_dg_q, action_gt_q, action_es])          # The action needed to be tested and corrected, [UPG_p, UPG_q, DG_p, GT_p, DG_q, GT_q, STORAGE]

        ## hyperparameter
        n_bus = 6
        balance_node = 1
        p_factor = 0.95
        p_tran_q = np.tan(np.arccos(p_factor))
        BASE_P = 100                   #用于求标幺值的基准功率
        BASE_Q = BASE_P * p_tran_q
        V_REF = 1
        maxp_storage = self.device_config['pso'][4]
        capacity_storage = self.device_config['pso'][5]
        efficiency = self.device_config['pso'][6]
        ubp_storage = min(capacity_storage * SOC *efficiency, maxp_storage)
        lbp_storage = max((SOC - 1) * capacity_storage /efficiency, -maxp_storage)
        g_list = np.array([[-999, 999],         #generator operation bound: upg
                        [0, self.device_config['pso'][0]],          #DG_p
                        [0, self.device_config['pso'][1]],          #GT_p
                        [0, self.device_config['pso'][2]],          #DG_q
                        [0, self.device_config['pso'][3]],          #GT_q
                        [lbp_storage, ubp_storage]])          #Storage

        br_p = np.array([[1, 2, 0.017, 0.017],         #line reactance matrix: from_node, to_node, R, X
                         [1, 4, 0.0258, 0.0258],
                         [2, 3, 0.0197, 0.0197],
                         [2, 6, 0.018, 0.018],
                         [3, 6, 0.037, 0.037],
                         [4, 5, 0.037, 0.037],
                         [5, 6, 0.014, 0.014]])

        br_flow_constr = np.array([[1, 2, 200],         #line flow constraints
                                    [1, 4, 200],
                                    [2, 3, 200],
                                    [2, 6, 200],
                                    [3, 6, 200],
                                    [4, 5, 200],
                                    [5, 6, 200]])

        max_line_flow = np.zeros((n_bus,n_bus))                 #line flow constraint matrix
        for branch in br_flow_constr:
            max_line_flow[branch[0]-1, branch[1]-1] = branch[2]
            max_line_flow[branch[1]-1, branch[0]-1] = branch[2]

        #Optimizer
        m = gp.Model('PowerNetworkCorrection')
        m.setParam('OutputFlag', 0)

        p_upg = m.addVar(lb=g_list[0][0], ub=g_list[0][1], name='power output of UPG')
        q_upg = m.addVar(lb=g_list[0][0], ub=g_list[0][1], name='reactive power output of UQG')
        p_g1 = m.addVar(lb=g_list[1][0], ub=g_list[1][1], name='power output of DG')
        p_g2 = m.addVar(lb=g_list[2][0], ub=g_list[2][1], name='power output of GT')  
        q_g1 = m.addVar(lb=g_list[3][0], ub=g_list[3][1], name='power output of DG')
        q_g2 = m.addVar(lb=g_list[4][0], ub=g_list[4][1], name='power output of GT')
        p_storage = m.addVar(lb=g_list[5][0], ub=g_list[5][1], name='power output of Storage')  

        v_node = {}
        for i in range(n_bus):
            v_node[i] = m.addVar(lb=0.9, ub = 1.05, name='voltage angle from bus 1 to bus 6')
        v_node[balance_node] = V_REF
        P_ex = m.addVars(n_bus,n_bus, lb=-999, ub=999, name='power exchange from bus 1 to bus 6')
        Q_ex = m.addVars(n_bus,n_bus, lb=-999, ub=999, name='reactive power exchange from bus 1 to bus 6')

        # 重组数据
        p_g_set = np.array([p_g1, 0, 0, 0, p_g2, 0])
        q_g_set = np.array([q_g1, 0, 0, 0, q_g2, 0])
        p_upg_set = np.array([p_upg, 0, 0, 0, 0, 0])
        q_upg_set = np.array([q_upg, 0, 0, 0, 0, 0])
        p_wt_set = np.array([0, 0, wt_output, 0, 0, 0])
        p_pv_set = np.array([0, pv_output, 0, 0, 0, 0])
        p_storage_set = np.array([0, p_storage, 0, 0, 0, 0])


        # 生成邻接矩阵
        R = self.generateRmatrix(n_bus, br_p) 
        X = self.generateXmatrix(n_bus, br_p)     
        C = self.generateCmatrix(n_bus, br_p)       

        for i in range(n_bus):
            m.addConstr((p_g_set[i] + p_upg_set[i] + p_wt_set[i] + p_pv_set[i] + p_storage_set[i])
                        == el[i] + gp.quicksum(P_ex[i,j]*C[i,j] for j in range(n_bus)))
            
            m.addConstr(q_g_set[i] + q_upg_set[i] == el_q[i] + gp.quicksum(Q_ex[i,j]*C[i,j] for j in range(n_bus)))
            
        for i in range(n_bus):
            for j in range(n_bus):
                m.addConstr((v_node[j] - v_node[i]) * C[i,j] == - (R[i,j]*P_ex[i,j]/BASE_P + X[i,j]*Q_ex[i,j]/BASE_Q)/V_REF)


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
        m.setObjective(gp.quicksum(([p_g1, p_g2, q_g1, q_g2, p_storage][i-2] - action_p[i]) ** 2 for i in range(2,7)), GRB.MINIMIZE)    #目标方程不能使用numpy L2-norm

        m.optimize()

        epsilon = 0.0001

        # if m.objVal <= epsilon:
        #     action_safe = action_p
        # else:
        #     generator_output = [p_upg.X, p_g1.X, p_storage.X, p_g2.X]
        #     action_safe = np.array(generator_output)

        OV = m.objVal
        generator_output = [p_upg.X, q_upg.X, p_g1.X, p_g2.X, q_g1.X, q_g2.X, p_storage.X]
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
        gl[1] += load_gt
        gl[3] += load_gb

        #Action formulation
        action_gw1 = self.action[5] * self.device_config['gso'][0]
        action_gw2 = self.action[6] * self.device_config['gso'][1]
        action_gs = - self.action[7] * self.device_config['gso'][2]
        action_ugg = gl.sum() - action_gw1 - action_gw2 - action_gs

        action_p = np.array([action_ugg, action_gw1, action_gw2, action_gs])          # The action needed to be tested and corrected, [UGG, GW1, GW2, STORAGE]

        ## hyperparameter
        n_node_gso = 7
        n_pipeline = 6
        balance_node = 1
        lambda_l = 2                                               #取值范围一般为1.5-4
        capacity_storage = self.device_config['gso'][3]
        maxp_storage = self.device_config['gso'][2]
        efficiency = self.device_config['gso'][4]
        ubp_storage = min(capacity_storage * SOC *efficiency, maxp_storage)
        lbp_storage = max((SOC - 1) * capacity_storage /efficiency, -maxp_storage)

        g_list = np.array([[0, 999],         #generator operation bound: ugg
                        [0, self.device_config['gso'][0]],          #GW1
                        [0, self.device_config['gso'][1]],            #GW2
                        [lbp_storage, ubp_storage]])          #Storage

        pipeline_flow_constr = np.array([[1, 2, 400],         #line flow constraints
                                    [2, 4, 200],
                                    [4, 7, 200],
                                    [2, 5, 200],
                                    [3, 5, 200],
                                    [5, 6, 200]])
        
        pipeline_para = np.array([[1, 2, 1, 9.3960, 4],         #line parameter: from node, to node, node type(1-active;0-inactive), C_mn, length (km)
                          [2, 4, 1, 3.7584, 10],
                          [4, 7, 0, 6.2640, 6],
                          [2, 5, 1, 0.9396, 40],
                          [3, 5, 0, 6.2640, 6],
                          [5, 6, 0, 6.2640, 6]])

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
        f_storage = m.addVar(lb=g_list[3][0], ub=g_list[3][1], name='power output of Storage')        

        f_pipeline = m.addVars(n_pipeline, lb=0, ub=pipeline_flow_constr[5,2], name='gas flow of pipeline')  # f_pipeline 1-7 的建模
        d_pipeline = m.addVars(n_pipeline, vtype=GRB.BINARY)

        p_node = {}
        p_node_sq = {}
        for i in range(n_node_gso):
            p_node[i] = m.addVar(lb=30, ub=60, name='gas pressure of node')
            p_node_sq[i] = m.addVar(lb=900, ub=3600, name='gas pressure of node')
        p_node[balance_node] = 45
        p_node_sq[balance_node] = 45*45
        
        # 重组数据
        f_gw_set = np.array([0, 0, 0, 0, 0, f_gw2, f_gw1])
        f_ugg_set = np.array([f_ugg, 0, 0, 0, 0, 0, 0])
        f_storage_set = np.array([0, 0, 0, 0, f_storage, 0, 0])
        f_pipeline_set = np.array([-f_pipeline[0]*(2*d_pipeline[0]-1),
                                    f_pipeline[0]*(2*d_pipeline[0]-1) - f_pipeline[1]*(2*d_pipeline[1]-1) - f_pipeline[3]*(2*d_pipeline[3]-1),
                                    -f_pipeline[4]*(2*d_pipeline[4]-1), 
                                    f_pipeline[1]*(2*d_pipeline[1]-1) - f_pipeline[2]*(2*d_pipeline[2]-1),
                                    f_pipeline[3]*(2*d_pipeline[3]-1) + f_pipeline[4]*(2*d_pipeline[4]-1) - f_pipeline[5]*(2*d_pipeline[5]-1),
                                    f_pipeline[5]*(2*d_pipeline[5]-1),
                                    f_pipeline[2]*(2*d_pipeline[2]-1)
                                    ])

        # 网络约束
        for i in range(n_pipeline):
            if pipeline_para[i,2] == 0:
                fn = pipeline_para[i,0] - 1
                tn = pipeline_para[i,1] - 1
                u = gp.QuadExpr(2*d_pipeline[i]*p_node_sq[fn])
                v = gp.QuadExpr(2*d_pipeline[i]*p_node_sq[tn])
                w = gp.QuadExpr(f_pipeline[i]*f_pipeline[i])
                m.addConstr( w - pipeline_para[i,3]*(u - p_node_sq[fn] - v + p_node_sq[tn]) == 0 )                        # relationship constraints between gas flow and nodal gas pressure in each inactive gas pipelineline 
            else:
                fn = pipeline_para[i,0] - 1
                tn = pipeline_para[i,1] - 1
                m.addConstr( d_pipeline[i]*p_node[fn] <= d_pipeline[i]*p_node[tn])
                m.addConstr( d_pipeline[i]*p_node[tn] <= d_pipeline[i]*lambda_l * p_node[fn])
                m.addConstr( (1-d_pipeline[i])*p_node[tn] <= (1-d_pipeline[i])*p_node[fn])
                m.addConstr( (1-d_pipeline[i])*p_node[fn] <= (1-d_pipeline[i])*lambda_l * p_node[tn])
        
        for i in range(n_node_gso):
            m.addConstr(p_node_sq[i] == p_node[i] * p_node[i])
        
        m.addConstrs(
            f_ugg_set[i] + f_gw_set[i] + f_storage_set[i] + f_pipeline_set[i] == gl[i] for i in range(n_node_gso)
            )                                     # nodal gas balance constraints

        # 优化目标方程
        m.setObjective(gp.quicksum(([f_gw1, f_gw2, f_storage][i-1] - action_p[i]) ** 2 for i in range(1,4)), GRB.MINIMIZE) 

        #Optimization Result
        m.params.NonConvex = 2
        m.optimize()

        # Action Correction
        epsilon = 0.0001

        # if m.objVal <= epsilon:
        #     action_safe = action_p
        # else:
        #     generator_output = [f_ugg.X, f_gw1.X, f_gw2.X, f_storage.X]
        #     action_safe = np.array(generator_output)
        
        OV = m.objVal
        generator_output = [f_ugg.X, f_gw1.X, f_gw2.X, f_storage.X]
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
    
    def hso_correction(self):
        
        ## inputs
        hl = self.heat_demand    # heat load / kJ
        SOC = self.SOC_heat
               
        t_amb = 10           # ！后续采取变化的日内室外温度！  

        # Action formulation
        # action_ehp = self.action[0] * self.device_config['hso'][0]
        # action_gb = self.action[1] * self.device_config['hso'][1]
        action_ehp = self.action[8]
        action_gb = self.action[9]
        action_ts = - self.action[10] * self.device_config['hso'][2]

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

        lb_ts = 65
        ub_ts = 100
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
        
        ObjVal = []
        action_hso, ov1, temp_supply, temp_return = self.hso_correction()
        ObjVal.append(ov1)

        load_ehp = action_hso[0] / 3                         #考虑能源转换效率
        action_pso, ov2, pso_flow_p, pso_flow_q, Voltage = self.pso_correction(load_ehp)
        ObjVal.append(ov2)

        load_gt = action_pso[3] / 2                          #考虑能源转换效率
        load_gb = action_hso[1] / 5                          #考虑能源转换效率
        action_gso, ov3, gso_flow, Pressure = self.gso_correction(load_gt, load_gb)
        ObjVal.append(ov3)

        # action conversion back into the primal form
        action_safe = np.zeros(11) 
        action_safe[0] = action_pso[2] / self.device_config['pso'][0]
        action_safe[1] = action_pso[3] / self.device_config['pso'][1]
        action_safe[2] = action_pso[4] / self.device_config['pso'][2]
        action_safe[3] = action_pso[5] / self.device_config['pso'][3]
        action_safe[4] = - (action_pso[6] / self.device_config['pso'][4])

        action_safe[5] = action_gso[1] / self.device_config['gso'][0]
        action_safe[6] = action_gso[2] / self.device_config['gso'][1]
        action_safe[7] = - (action_gso[3] / self.device_config['gso'][2])

        action_safe[8] = action_hso[0] / self.device_config['hso'][0]
        action_safe[9] = action_hso[1] / self.device_config['hso'][1]
        action_safe[10] = - (action_hso[2] / self.device_config['hso'][2])

        main_grid_interaction = np.zeros(3)                    # [upg_p, upg_p, ugg]
        main_grid_interaction[0] = action_pso[0]
        main_grid_interaction[1] = action_pso[1]
        main_grid_interaction[2] = action_gso[0]

        if self.test:
            return action_safe, main_grid_interaction, ObjVal, pso_flow_p, pso_flow_q, Voltage, gso_flow, Pressure, temp_supply, temp_return
        else:
            return action_safe, main_grid_interaction, ObjVal






