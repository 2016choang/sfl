
import numpy as np

from rlpyt.samplers.collectors import BaseEvalCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args

# For sampling, serial sampler can use Cpu collectors.


class SerialEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.eval_mode(itr)
        self.agent.reset()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        return completed_traj_infos

class SerialLandmarksEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        self.env_positions = np.full((len(self.envs), len(self.envs[0].agent_pos)), -1, dtype=int)
        self.eval_trajectories = [[] for _ in range(len(self.envs))]
        for b, env in enumerate(self.envs):
            observations.append(env.reset())
            self.env_positions[b] = env.agent_pos
            self.eval_trajectories[b].append(env.agent_pos)
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt, self.env_positions)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                self.env_positions[b] = env.agent_pos
                if len(completed_traj_infos) < 1:
                    self.eval_trajectories[b].append(env.agent_pos)
                if d:
                    self.agent.log_eval(b, self.env_positions[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                    self.env_positions[b] = env.agent_pos
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        self.eval_trajectories = np.array(self.eval_trajectories)
        return completed_traj_infos


class SerialVizdoomEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            eval_settings,
            trajectories_per_setting,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        self.env_positions = np.full((len(self.envs), len(self.envs[0].agent_pos)), -1, dtype=int)
        self.eval_trajectories = [[] for _ in range(len(self.envs))]
        eval_settings_queue = [setting for setting in self.eval_settings for _ in self.trajectories_per_setting]
        for b, env in enumerate(self.envs):
            name, goal_distance_range, step_budget = eval_settings_queue.pop()
            env.name = name
            goal_state = env.sample_state_from_point(goal_distance_range)
            env.set_goal_state(goal_state)
            self.agent.update_eval_goal(goal_state)
            env.step_budget = step_budget
            observations.append(env.reset())
            self.env_positions[b] = env.agent_pos
            self.eval_trajectories[b].append(env.agent_pos)
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        finished = False
        while True:
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt, self.env_positions)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                self.env_positions[b] = env.agent_pos
                if len(completed_traj_infos) < 1:
                    self.eval_trajectories[b].append(env.agent_pos)
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if d:
                    self.agent.log_eval(b, self.env_positions[b])
                    completed_traj_infos.append([env.name,
                                                 traj_infos[b].terminate(o)])
                    if not eval_settings_queue:
                        finished = True
                        break
                    name, goal_distance_range, step_budget = eval_settings_queue.pop()
                    env.name = name
                    goal_state = env.sample_state_from_point(goal_distance_range)
                    env.set_goal_state(goal_state)
                    self.agent.update_eval_goal(goal_state)
                    env.step_budget = step_budget
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                    self.env_positions[b] = env.agent_pos
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if finished:
                break
        self.eval_trajectories = np.array(self.eval_trajectories)
        return completed_traj_infos
