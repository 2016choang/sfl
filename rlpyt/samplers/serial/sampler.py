import json

from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.utils.logging import logger
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector


class SerialSampler(BaseSampler):
    """Uses same functionality as ParallelSampler but does not fork worker
    processes; can be easier for debugging (e.g. breakpoint() in master).  Use
    with collectors which sample actions themselves (e.g. under cpu
    category)."""

    def __init__(self, *args, CollectorCls=CpuResetCollector,
            eval_CollectorCls=SerialEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        return self.eval_collector.collect_evaluation(itr)


class VizdoomSampler(BaseSampler):
    """Uses same functionality as ParallelSampler but does not fork worker
    processes; can be easier for debugging (e.g. breakpoint() in master).  Use
    with collectors which sample actions themselves (e.g. under cpu
    category)."""

    def __init__(self, *args, eval_settings, start_goal_pairs_path, video_data_save_file,
            trajectories_per_setting,
            CollectorCls=CpuResetCollector,
            eval_CollectorCls=SerialEvalCollector, **kwargs):
        self.eval_settings = eval_settings
        with open(start_goal_pairs_path, 'r') as f:
            start_goal_pairs = json.load(f)
            self.start_goal_pairs = {name: pairs[:trajectories_per_setting] \
                for name, pairs in start_goal_pairs.items()}
        self.video_data_save_file = video_data_save_file
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                eval_settings=self.eval_settings,
                start_goal_pairs=self.start_goal_pairs,
                video_data_save_file=self.video_data_save_file
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Vizdoom Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        return self.eval_collector.collect_evaluation(itr) 
                    