import copy
from collections import deque, defaultdict
import io
import psutil
import time
import torch
import math
import numpy as np
import os
from scipy.stats import entropy

import matplotlib.pyplot as plt
import networkx as nx
import PIL.Image
from torchvision.transforms import ToTensor

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


class MinibatchRlBase(BaseRunner):

    _eval = False

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            seed=None,
            affinity=None,
            log_interval_steps=1e5,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())

    def startup(self):
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))

    def get_n_itr(self):
        log_interval_itrs = max(self.log_interval_steps //
            self.itr_batch_size, 1)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._cum_completed_trajs = 0
        self._last_update_counter = 0

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            cum_steps=itr * self.sampler.batch_size * self.world_size,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0):
        if itr > 0:
            self.pbar.stop()
        self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.world_size *
            self.log_interval_itrs)
        updates_per_second = (float('nan') if itr == 0 else
            new_updates / train_time_elapsed)
        samples_per_second = (float('nan') if itr == 0 else
            new_samples / train_time_elapsed)
        replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
            new_samples)
        cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
            ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

        if self._eval:
            logger.record_tabular('CumTrainTime',
                self._cum_time - self._cum_eval_time)  # Already added new eval_time.
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumTime (s)', self._cum_time)
        logger.record_tabular('CumSteps', cum_steps)
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('CumUpdates', self.algo.update_counter)
        logger.record_tabular('StepsPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        logger.record_tabular('ReplayRatio', replay_ratio)
        logger.record_tabular('CumReplayRatio', cum_replay_ratio)
        self._log_infos(traj_infos, itr)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, traj_infos=None, itr=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                        [info[k] for info in traj_infos], itr)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v, itr)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRl(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance online using learning
    trajectories."""

    def __init__(self, log_traj_window=100, **kwargs):
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr):
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
            sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr)
        self._new_completed_trajs = 0


class MinibatchRlEval(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance offline using evaluation
    trajectories."""

    _eval = True

    def train(self):
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                logger.log_itr_info(itr, opt_info)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)

                    # summary_writer = logger.get_tf_summary_writer()
                    # # Debugging layer parameters
                    # for name, param in self.agent.model.named_parameters():
                    #     if param.requires_grad:
                    #         summary_writer.add_histogram(name, param.flatten(), itr)

        self.shutdown()

    def evaluate_agent(self, itr):
        if itr > 0:
            self.pbar.stop()
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler.
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def initialize_logging(self):
        super().initialize_logging()
        self._cum_eval_time = 0

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))
        self._cum_eval_time += eval_time
        logger.record_tabular('CumEvalTime', self._cum_eval_time)
        super().log_diagnostics(itr, eval_traj_infos, eval_time)


def save_image(title, itr):
    plt.title(title)
    summary_writer = logger.get_tf_summary_writer()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf).convert('RGB')
    image = ToTensor()(image)
    summary_writer.add_image(title, image, itr)


class MinibatchDSREval(MinibatchRlEval):
    _eval = True

    def __init__(self, log_dsr_interval_steps=1e4, **kwargs):
        self.log_dsr_interval_steps = int(log_dsr_interval_steps)
        super().__init__(**kwargs)

    def train(self):
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                logger.log_itr_info(itr, opt_info)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
                    self.algo.update_scheduler(self._opt_infos)

                if (itr + 1) % self.log_dsr_interval_steps == 0:
                    self.log_dsr(itr)
                    
        self.shutdown()
        
    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        env = self.sampler.collector.envs[0]
        state_entropy = entropy(env.visited.flatten(), base=2)
        if not np.isnan(state_entropy):
            logger.record_tabular_stat('Entropy', state_entropy, itr)
        # if self.agent.landmarks is not None:
        #     logger.record_tabular_stat('Landmark Threshold', self.agent.landmarks.threshold, itr)

    def log_dsr(self, itr):
        env = self.sampler.collector.envs[0]
        
        figure = plt.figure(figsize=(7, 7))
        plt.imshow(env.render(8))
        save_image('Environment', itr)

        figure = plt.figure(figsize=(7, 7))
        plt.imshow(env.visited.T)
        circle = plt.Circle(tuple(env.start_pos), 0.2, color='r')
        plt.gca().add_artist(circle)
        # circle = plt.Circle(tuple(env.goal_pos), 0.2, color='purple')
        # plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('State Visitation Heatmap', itr)

        env_kwargs = self.sampler.env_kwargs
        env_kwargs['minigrid_config']['epsilon'] = 0.0
        dsr_env = self.sampler.EnvCls(**env_kwargs)
        dsr_env.reset()

        dsr = self.agent.get_dsr(dsr_env)
        torch.save(dsr, os.path.join(logger.get_snapshot_dir(), 'dsr_itr_{}.pt'.format(itr)))
        subgoal = tuple(env.start_pos)

        figure = plt.figure(figsize=(7, 7))
        dsr_heatmap = self.agent.get_dsr_heatmap(dsr, subgoal=subgoal)
        plt.imshow(dsr_heatmap.T)
        circle = plt.Circle(subgoal, 0.2, color='r')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('Cosine Similarity in SF Space', itr)

        figure = plt.figure(figsize=(7, 7))
        q_values = self.agent.get_q_values(dsr_env, dsr, subgoal=subgoal)
        plt.imshow(q_values.max(axis=2).T)
        for x in range(q_values.shape[0]):
            plt.axvline(x + 0.5, color='k', linestyle=':')
            for y in range(q_values.shape[1]):
                plt.axhline(y + 0.5, color='k', linestyle=':')
                
                if (x, y) == subgoal:
                    circle = plt.Circle((x, y), 0.2, color='r')
                    plt.gca().add_artist(circle)
                
                else:
                    if any(np.isnan(q_values[x, y])):
                        continue

                    action = q_values[x, y].argmax()
                    dx = 0
                    dy = 0
                    if action == 0:
                        dx = 0.35
                    elif action == 1:
                        dy = 0.35
                    elif action == 2:
                        dx = -0.35
                    else:
                        dy = -0.35

                    plt.arrow(x - dx, y - dy, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
        plt.colorbar()
        save_image('Subgoal Policy', itr)


class MinibatchLandmarkDSREval(MinibatchDSREval):
    _eval = True

    def __init__(self, min_steps_landmark=2e4, log_landmark_steps=1e4, **kwargs):
        self.min_steps_landmark = int(min_steps_landmark)
        self.log_landmark_steps = int(log_landmark_steps)
        super().__init__(**kwargs)

    def train(self):
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        self.agent.set_env_true_dist(self.sampler.collector.envs[0])
        self.agent.set_oracle_landmarks(self.sampler.eval_collector.envs[0])
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                if not self.agent.explore:
                    samples = copy.deepcopy(samples)
                    samples.env.done[-1] = True
                    traj_infos = copy.deepcopy(traj_infos)
                    while not self.agent.explore:
                        # TODO: Make sampler function to simply move in environment
                        self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                logger.log_itr_info(itr, opt_info)
                self.store_diagnostics(itr, traj_infos, opt_info)
                
                if (itr + 1) >= self.min_steps_landmark:
                    if self.agent.landmarks is None:
                        goal_obs = self.sampler.eval_collector.envs[0].get_goal_state()
                        self.agent.create_landmarks(goal_obs)
                    self.agent.update_landmarks(itr)

                landmarks_eval = False
                if (itr + 1) % self.log_interval_itrs == 0:
                    landmarks_eval = self.agent.enter_eval_mode()
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
                    self.algo.update_scheduler(self._opt_infos)

                if (itr + 1) % self.log_dsr_interval_steps == 0:
                    self.log_dsr(itr)
                    plt.close('all')

                if (itr + 1) % self.log_landmark_steps == 0:
                    self.log_landmarks(itr)
                    plt.close('all')

                if landmarks_eval:
                    summary_writer = logger.get_tf_summary_writer()                   
                    summary_writer.add_text("Path to goal", ','.join(map(str, self.agent.path)), itr)
                    logger.record_tabular_stat('EndDistanceToGoal', np.average(self.agent.eval_distances), itr)

                    eval_env = self.sampler.eval_collector.envs[0]
                    eval_grid = eval_env.visited.T.copy()

                    figure = plt.figure(figsize=(7, 7))
                    plt.imshow(eval_grid)
                    for pos, landmarks in self.agent.eval_end_pos.items():
                        plt.text(pos[0] -0.25, pos[1] + 0.25, ','.join(map(str, landmarks)), fontsize=6)
                    circle = plt.Circle(tuple(eval_env.start_pos), 0.2, color='r')
                    plt.gca().add_artist(circle)
                    circle = plt.Circle(tuple(eval_env.goal_pos), 0.2, color='purple')
                    plt.gca().add_artist(circle)
                    plt.colorbar()
                    save_image('Eval visitations and end positions', itr)

                    self.agent.exit_eval_mode()


        self.shutdown()

    @torch.no_grad()
    def log_landmarks(self, itr):
        # If no landmark info to log
        if self.agent.landmarks is None or self.agent.landmarks.num_landmarks == 0:
            return

        # landmark_reach_percentage = self.agent.landmark_reaches / np.clip(self.agent.landmark_attempts, 1, None)
        # ind = np.arange(len(landmark_reach_percentage))
        # plt.bar(ind, landmark_reach_percentage)

        figure = plt.figure(figsize=(7, 7))
        landmark_true_reach_percentage = self.agent.landmark_true_reaches / np.clip(self.agent.landmark_attempts, 1, None)
        
        ind = np.arange(len(landmark_true_reach_percentage))
        plt.bar(ind, landmark_true_reach_percentage, label='True distance reach')
        plt.xlabel('ith Landmark')
        plt.ylabel('Reach Percentage')
        plt.legend()
        save_image('Landmarks reach rates', itr)

        figure = plt.figure(figsize=(7, 7))
        landmark_dist_completed = self.agent.landmark_dist_completed
        for i, progress in enumerate(landmark_dist_completed):
            if progress:
                landmark_dist_completed[i] = np.average(progress)
            else:
                landmark_dist_completed[i] = 0
        
        ind = np.arange(len(landmark_dist_completed))
        plt.bar(ind, landmark_dist_completed)
        plt.xlabel('ith Landmark')
        plt.ylabel('End/Start Distance Ratio')
        save_image('Landmarks end-start distance ratio', itr)

        if self.agent.goal_landmark_dist_completed:
            logger.record_tabular_stat('End-StartDistanceRatio', np.average(self.agent.goal_landmark_dist_completed), itr)

        logger.record_tabular_stat('LandmarksAdded', self.agent.landmarks.landmark_adds, itr)
        logger.record_tabular_stat('LandmarksRemoved', self.agent.landmarks.landmark_removes, itr)

        if self.agent.total_landmark_paths:
            logger.record_tabular_stat('CorrectStartLandmarkRatio', self.agent.correct_start_landmark / self.agent.total_landmark_paths, itr)

        if self.agent.dist_ratio_start_landmark:
            logger.record_tabular_stat('True-EstimateDistanceStartLandmarkRatio', np.average(self.agent.dist_ratio_start_landmark), itr)

        if self.agent.landmarks.success_rates:
            logger.record_tabular_stat('LandmarkSuccessRate', np.average(self.agent.landmarks.success_rates), itr)
            logger.record_tabular_stat('NonZeroLandmarkSuccessRate', np.average(self.agent.landmarks.non_zero_success_rates), itr)
            logger.record_tabular_stat('ZeroSuccessEdgeRatio', np.average(self.agent.landmarks.zero_success_edge_ratio), itr)

        self.agent.reset_logging()

        figure = plt.figure(figsize=(7, 7))
        visitations = self.agent.landmarks.visitations
        plt.bar(np.arange(len(visitations)), visitations)
        plt.xlabel('Landmark')
        plt.ylabel('Visitations')
        save_image('Landmark visitation counts', itr)

        env = self.sampler.collector.envs[0]
        landmarks_grid = env.visited.T.copy()
        landmarks_grid[landmarks_grid == 0] = -1
        landmarks_grid[landmarks_grid != -1] = 0
        
        node_labels = defaultdict(list)

        for i, observation in enumerate(self.agent.landmarks.observations):
            diff = observation[:, :, 0] - observation[:, :, 2]
            idx = diff.argmax().detach().cpu().numpy()
            pos = (idx // 25, idx % 25)
            node_labels[pos].append(i)
            landmarks_grid[pos] += 1

        figure = plt.figure(figsize=(7, 7))
        plt.imshow(landmarks_grid)
        for pos, nodes in node_labels.items():
            plt.text(pos[1] -0.25, pos[0] + 0.25, ','.join(map(str, nodes)), fontsize=6)
        circle = plt.Circle(tuple(env.start_pos), 0.2, color='r')
        plt.gca().add_artist(circle)
        circle = plt.Circle(tuple(env.goal_pos), 0.2, color='purple')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('Landmarks', itr)

        figure = plt.figure(figsize=(7, 7))
        self.agent.generate_graph()

        G = self.agent.landmarks.graph
        pos = nx.circular_layout(G)

        non_zero_edges = {}
        zero_edges = {}

        for index, edge_info in dict(G.edges).items():
            if self.agent.landmarks.zero_edge_indices is not None and self.index in self.agent.landmarks.zero_edge_indices:
                zero_edges[index] = edge_info
            else:
                non_zero_edges[index] = edge_info
        
        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_edges(G, pos, edgelist=non_zero_edges, width=6, edge_color='black')
        nx.draw_networkx_edges(G, pos, edgelist=zero_edges, width=6, edge_color='red')

        edge_labels = nx.get_edge_attributes(G, 'weight')
        for k, v in edge_labels.items():
            edge_labels[k] = round(v, 3)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        nx.draw_networkx_edge_labels(G, pos, font_size=10, font_family='sans-serif', edge_labels=edge_labels)
        plt.axis('off')
        save_image('Landmarks graph', itr)

    def evaluate_agent(self, itr):
        if itr > 0:
            self.pbar.stop()
        logger.log("Evaluating agent...")
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time
