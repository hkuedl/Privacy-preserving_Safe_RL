import torch
from Environment.Env_con_hhL_NewObs import Multi_Energy_Env_hhl
from Environment.Env_con_mhL import Multi_Energy_Env_mhl
from Environment.Env_con_lhL import Multi_Energy_Env_lhl

import Simulation.utils.pytorch_util as ptu
from Simulation.Replay_buffer import ReplayBuffer
from Simulation.utils.policy import TanhGaussianPolicy
from Simulation.SAC import SACTrainer
from Simulation.utils.networks import ConcatMlp
# from Simulation.Algorithm import BatchRLAlgorithm                 # 11个月一循环
from Environment.Env_NewObs_Oneday_PSO_HSO_GSO_AC import Multi_Energy_Env
from Simulation.Algorithm_day_OneOp_PSO_HSO_GSO_Privacy_AC import BatchRLAlgorithm               #一天一循环

 

def experiment(variant):
    env =  Multi_Energy_Env()

    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()

    device = torch.device("cuda:2")

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        observation_dim = obs_dim,
        action_dim=action_dim
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = BatchRLAlgorithm(
        trainer=trainer,
        env = env,
        policy = policy,
        replay_buffer = replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",

        layer_size=256,
        replay_buffer_size=int(1224000),
        algorithm_kwargs=dict(
            num_epochs=51000,
            num_env_steps_per_epoch=24,                    #一天为一个episode，一天24小时，最后一个小时无转移
            num_trains_per_train_epoch=50,
            min_num_epoch_before_training=668,
            batch_size=4096,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=6E-4,
            qf_lr=6E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )

    ptu.set_gpu_mode(True, gpu_id=2)  # optionally set the GPU (default=False)
    experiment(variant)

