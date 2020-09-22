import copy
from collections import deque, defaultdict
import io
import os
import psutil
import re
import time
import math

from matplotlib import cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL.Image
from scipy.special import softmax
from scipy.stats import entropy
import torch
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
            logger.record_tabular_stat('CumTrainTime',
                self._cum_time - self._cum_eval_time, itr)  # Already added new eval_time.
        logger.record_tabular('Iteration', itr)
        logger.record_tabular_stat('CumTime (s)', self._cum_time, itr)
        logger.record_tabular_stat('CumSteps', cum_steps, itr)
        logger.record_tabular_stat('CumCompletedTrajs', self._cum_completed_trajs, itr)
        logger.record_tabular_stat('CumUpdates', self.algo.update_counter, itr)
        logger.record_tabular_stat('StepsPerSecond', samples_per_second, itr)
        logger.record_tabular_stat('UpdatesPerSecond', updates_per_second, itr)
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
        logger.record_tabular_stat('CumEvalTime', self._cum_eval_time, itr)
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

    def __init__(self,
                 log_dsr_interval_steps=1e4,
                 **kwargs):
        save__init__args(locals())
        super().__init__(**kwargs)
    
    def get_n_itr(self):
        n_itr = super().get_n_itr()
        self.log_dsr_interval_itrs = max(int(self.log_dsr_interval_steps // self.itr_batch_size), 1)
        return n_itr

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

                if (itr + 1) % self.log_dsr_interval_itrs == 0:
                    self.log_dsr(itr)
                    
        self.shutdown()
        
    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        env = self.sampler.collector.envs[0]
        state_entropy = entropy(env.visited.flatten(), base=2)
        if not np.isnan(state_entropy):
            logger.record_tabular_stat('Entropy', state_entropy, itr)

    def log_dsr(self, itr):
        # 1. Render actual environemnt
        env = self.sampler.collector.envs[0]
        figure = plt.figure(figsize=(7, 7))
        plt.imshow(env.render(8))
        save_image('Environment', itr)
        plt.close()

        # 2. Heatmap of state vistations during training
        figure = plt.figure(figsize=(7, 7))
        plt.imshow(env.visited.T)
        circle = plt.Circle(tuple(env.start_pos), 0.2, color='r')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('State Visitation Heatmap', itr)
        plt.close()

        # Retrieve feature and successor feature representations for all states
        env_kwargs = self.sampler.env_kwargs
        env_kwargs['minigrid_config']['epsilon'] = 0.0
        dsr_env = self.sampler.EnvCls(**env_kwargs)
        dsr_env.reset()

        features, dsr = self.agent.get_representations(dsr_env)
        torch.save(features, os.path.join(logger.get_snapshot_dir(), 'features_itr_{}.pt'.format(itr)))
        torch.save(dsr, os.path.join(logger.get_snapshot_dir(), 'dsr_itr_{}.pt'.format(itr)))
        subgoal = tuple(env.true_goal_pos)

        # 3. Distance visualization in feature space
        figure = plt.figure(figsize=(7, 7))
        feature_heatmap = self.agent.get_representation_heatmap(features, subgoal=subgoal, mean_axes=2)
        plt.imshow(feature_heatmap.T)
        circle = plt.Circle(subgoal, 0.2, color='r')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('Cosine Similarity in Feature Space', itr)
        plt.close()

        # 4. T-SNE Visualization of features
        num_rooms = len(dsr_env.rooms) + 1 if hasattr(dsr_env, 'rooms') else 1

        feature_tsne, rooms = self.agent.get_tsne(dsr_env, features, mean_axes=2)
        colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, num_rooms)]
        
        figure = plt.figure(figsize=(7, 7))
        tsne_data = feature_tsne[rooms == 0]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], label='Doorway', marker='*', color=colors[0])
        for i in range(1, num_rooms):
            tsne_data = feature_tsne[rooms == i]
            plt.scatter(tsne_data[:, 0], tsne_data[:, 1], label='Room ' + str(i), color=colors[i])
        plt.legend()
        save_image('T-SNE Embeddings of Features', itr)
        plt.close()

        # 5. Distance visualization in SF space
        figure = plt.figure(figsize=(7, 7))
        dsr_heatmap = self.agent.get_representation_heatmap(dsr, subgoal=subgoal)
        plt.imshow(dsr_heatmap.T)
        circle = plt.Circle(subgoal, 0.2, color='r')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('Cosine Similarity in SF Space', itr)
        plt.close()

        # 6. T-SNE Visualization of SF
        dsr_tsne, rooms = self.agent.get_tsne(dsr_env, dsr, mean_axes=2)
        colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, num_rooms)]
        
        figure = plt.figure(figsize=(7, 7))
        tsne_data = dsr_tsne[rooms == 0]
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], label='Doorway', marker='*', color=colors[0])
        for i in range(1, num_rooms):
            tsne_data = dsr_tsne[rooms == i]
            plt.scatter(tsne_data[:, 0], tsne_data[:, 1], label='Room ' + str(i), color=colors[i])
        plt.legend()
        save_image('T-SNE Embeddings of SF', itr)
        plt.close()

        # 7. Visualization of Q-values (SF-based)
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
                    plt.text(x - 0.25, y + 0.25, str(action), fontsize=6)
        plt.colorbar()
        save_image('Subgoal Policy', itr)
        plt.close()

class MinibatchLandmarkDSREval(MinibatchDSREval):
    _eval = True

    def __init__(self,
                 max_steps_sample=None,
                 steps_sample_test=1e4,
                 test_set_size=6e3,
                 min_steps_landmark_mode=2e4,
                 update_landmark_representation_interval_steps=1e3,
                 update_landmark_graph_interval_steps=1e3,
                 log_feature_info_interval_steps=1e3,
                 log_landmarks_interval_steps=1e4,
                 **kwargs):
        save__init__args(locals())
        super().__init__(**kwargs)

    def get_n_itr(self):
        n_itr = super().get_n_itr()
        if self.max_steps_sample:
            self.max_itr_sample = max(int(self.max_steps_sample // self.itr_batch_size), 1)
        else:
            self.max_itr_sample = None
        self.itr_sample_test = max(int(self.steps_sample_test // self.itr_batch_size), 1)
        self.min_itr_landmark_mode = max(int(self.min_steps_landmark_mode // self.itr_batch_size), 1)
        self.update_landmark_representation_interval_itrs = max(int(self.update_landmark_representation_interval_steps // self.itr_batch_size), 1)
        self.update_landmark_graph_interval_itrs = max(int(self.update_landmark_graph_interval_steps // self.itr_batch_size), 1)
        self.log_feature_info_interval_itrs = max(int(self.log_feature_info_interval_steps // self.itr_batch_size), 1)
        self.log_landmarks_interval_itrs = max(int(self.log_landmarks_interval_steps // self.itr_batch_size), 1)
        return n_itr

    def startup(self):
        n_itr = super().startup()
        train_envs = len(self.sampler.collector.envs)
        eval_envs = len(self.sampler.eval_collector.envs)
        # oracle_distance_matrix = self.sampler.collector.envs[0].oracle_distance_matrix  
        lines = np.array([(l.x1, l.y1, l.x2, l.y2) for l in self.sampler.collector.envs[0].state.sectors[0].lines if l.is_blocking])
        self.agent.initialize_landmarks(train_envs, eval_envs, lines)
        return n_itr

    def train(self):
        n_itr = self.startup()

        # # Create test set
        # self.agent.sample_mode(1)
        # for _ in range(self.itr_sample_test):
        #     samples, traj_infos = self.sampler.obtain_samples(1)
        #     self.algo.append_test_feature_samples(samples)
        # self.algo.create_test_set(int(self.test_set_size))

        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
            # self.log_eval_features(0)
        
        # Main loop
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                
                if not self.agent.landmarks and itr >= self.min_itr_landmark_mode:
                    self.agent.landmarks.activate()
                    self.agent.reset()

                if not self.max_itr_sample or itr < self.max_itr_sample:
                    samples, traj_infos = self.sampler.obtain_samples(itr)
                    self.algo.append_feature_samples(samples)  # feature replay buffer (policy-agnostic)
                    self.algo.append_dsr_samples(samples)  # SF replay buffer (random)

                # Train agent's neural networks
                self.agent.train_mode(itr)
                opt_info, feature_info = self.algo.optimize_agent(itr)
                logger.log_itr_info(itr, opt_info)
                self.store_diagnostics(itr, traj_infos, opt_info)

                if (itr + 1) % self.log_feature_info_interval_itrs == 0:
                    self.log_feature_info(itr, feature_info)

                # Update landmark representation
                if (itr + 1) % self.update_landmark_representation_interval_itrs == 0:
                    self.agent.sample_mode(itr)
                    self.agent.update_landmark_representation(itr)
                
                # Update landmark graph
                if (itr + 1) % self.update_landmark_graph_interval_itrs == 0:
                    self.agent.sample_mode(itr)
                    self.agent.update_landmark_graph(itr)

                # Evaluate agent
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    # self.log_eval_features(itr)
                    if eval_traj_infos is not None:
                        self.log_diagnostics(itr, eval_traj_infos, eval_time)
                    self.algo.update_scheduler(self._opt_infos)
                    self.log_eval_landmarks(itr)

                # Log successor features information
                if (itr + 1) % self.log_dsr_interval_itrs == 0:
                    self.log_dsr(itr)

                # Log landmarks information
                if (itr + 1) % self.log_landmarks_interval_itrs == 0:
                    self.log_landmarks(itr)

        self.shutdown()

    def log_feature_info(self, itr, feature_info):
        pass

    def log_eval_features(self, itr):
        eval_loss, eval_opt_info = self.algo.eval_feature_loss()
        logger.record_tabular_stat('FeatureLossEval', np.average(eval_loss), itr)
        for key, value in eval_opt_info.items():
            logger.record_tabular_stat('{}Eval'.format(key), np.average(value), itr)

    def log_eval_landmarks(self, itr):
        # Log eval landmarks information
        if self.agent.landmarks:
            summary_writer = logger.get_tf_summary_writer()
            eval_path_str = '\n'.join(','.join(map(str, path)) + ' ({:.3f})'.format(self.agent.landmarks.path_p[i]) \
                for i, path in enumerate(self.agent.landmarks.possible_paths) if len(path) <= 20)
            summary_writer.add_text("Path to goal", eval_path_str, itr)
            logger.record_tabular_stat('EndDistanceToGoal', np.average(self.agent.landmarks.eval_distances), itr)

            eval_env = self.sampler.eval_collector.envs[0]
            eval_grid = eval_env.visited.T.copy()

            # State visitation heatmap in evaluation mode and
            # agent's end position after executing landmark mode
            figure = plt.figure(figsize=(7, 7))
            plt.imshow(eval_grid)
            for pos, landmark in self.agent.landmarks.eval_end_pos.items():
                plt.text(pos[0] - 0.25, pos[1] + 0.25, str(landmark), fontsize=6)
            circle = plt.Circle(tuple(eval_env.start_pos), 0.2, color='r')
            plt.gca().add_artist(circle)
            circle = plt.Circle(tuple(eval_env.true_goal_pos), 0.2, color='purple')
            plt.gca().add_artist(circle)
            plt.colorbar()
            save_image('Eval visitations and end positions', itr)
            plt.close()

    @torch.no_grad()
    def log_landmarks(self, itr):
        # If no landmark info to log
        if not self.agent.landmarks:
            return
        
        self.agent.sample_mode(itr)

        env = self.sampler.collector.envs[0]

        # Save landmarks data
        self.agent.landmarks.save(os.path.join(logger.get_snapshot_dir(), 'landmarks_itr_{}.npz'.format(itr)))

        # # 1. Reach rate of ith landmark
        # figure = plt.figure(figsize=(7, 7))
        # landmark_true_reach_percentage = self.agent.landmarks.landmark_true_reaches / np.clip(self.agent.landmarks.landmark_attempts, 1, None)
        # ind = np.arange(len(landmark_true_reach_percentage))
        # plt.bar(ind, landmark_true_reach_percentage)
        # plt.xlabel('ith Landmark')
        # plt.ylabel('Reach Percentage')
        # plt.legend()
        # save_image('Landmarks reach rates', itr)
        # plt.close()

        # # 2. Percent distance covered to ith landmark
        # figure = plt.figure(figsize=(7, 7))
        # landmark_dist_completed = self.agent.landmark_dist_completed
        # for i, progress in enumerate(landmark_dist_completed):
        #     if progress:
        #         landmark_dist_completed[i] = np.average(progress)
        #     else:
        #         landmark_dist_completed[i] = 0
        # ind = np.arange(len(landmark_dist_completed))
        # plt.bar(ind, landmark_dist_completed)
        # plt.xlabel('ith Landmark')
        # plt.ylabel('Percent Distance Covered')
        # save_image('Landmarks percent distance covered', itr)

        # 3. Percent distance covered to goal landmark
        if self.agent.landmarks.goal_landmark_dist_completed:
            logger.record_tabular_stat('PercentDistanceCoveredToGoal',
                                       np.average(self.agent.landmarks.goal_landmark_dist_completed), itr)

        # 4. Statistics related to adding/removing landmarks
        logger.record_tabular_stat('LandmarksAdded', self.agent.landmarks.landmark_adds, itr)
        logger.record_tabular_stat('LandmarksRemoved', self.agent.landmarks.landmark_removes, itr)

        # 5. Percent of times we select correct starting landmark
        if self.agent.landmarks.total_landmark_paths:
            logger.record_tabular_stat('PercentCorrectStartLandmark',
                                       self.agent.landmarks.correct_start_landmark / self.agent.landmarks.total_landmark_paths, itr)

        # 6. Ratio of distance to correct start landmark over distance
        #    to estimated start landmark
        if self.agent.landmarks.dist_ratio_start_landmark:
            logger.record_tabular_stat('Correct-EstimatedStartLandmarkDistanceRatio',
                                       np.average(self.agent.landmarks.dist_ratio_start_landmark), itr)

        # 7. Statistics related to success rates of transitions between landmarks
        overall_success_rates = self.agent.landmarks.successes / np.clip(self.agent.landmarks.attempts, 1, None)
        interval_success_rates = self.agent.landmarks.interval_successes / np.clip(self.agent.landmarks.interval_attempts, 1, None)

        if hasattr(env, 'rooms'):
            oracle_edges = self.agent.landmarks.get_oracle_edges(env)
            figure = plt.figure(figsize=(7, 7))
            oracle_edges_matrix = np.maximum(oracle_edges, oracle_edges.T)
            G = nx.from_numpy_array(oracle_edges_matrix.astype(int), create_using=nx.DiGraph)
            pos = nx.circular_layout(G)

            nx.draw_networkx_nodes(G, pos, node_size=600)
            nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=2, edge_color='black')

            edge_labels = nx.get_edge_attributes(G, 'weight')
            for k, _ in edge_labels.items():
                if self.agent.landmarks.attempts[k] > 0:
                    edge_label = round(overall_success_rates[k], 3)
                else:
                    edge_label = -1
                edge_labels[k] = edge_label

            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            nx.draw_networkx_edge_labels(G, pos, font_size=6, font_family='sans-serif', edge_labels=edge_labels, connectionstyle='arc3,rad=0.1')
            plt.axis('off')
            save_image('Oracle landmarks graph', itr)
            plt.close()
        else:
            oracle_edges = True

        oracle_overall_success_rates = overall_success_rates[oracle_edges & (self.agent.landmarks.attempts > 0)]
        oracle_interval_success_rates = interval_success_rates[oracle_edges & (self.agent.landmarks.interval_attempts > 0)]
        logger.record_tabular_stat('OverallLandmarkSuccessRate',
                                   np.average(oracle_overall_success_rates), itr)
        logger.record_tabular_stat('IntervalLandmarkSuccessRate',
                                   np.average(oracle_interval_success_rates), itr)
        
        # 8. Visitation counts of landmarks
        figure = plt.figure(figsize=(7, 7))
        visitations = self.agent.landmarks.visitations
        plt.bar(np.arange(len(visitations)), visitations)
        plt.xlabel('Landmark')
        plt.ylabel('Visitations')
        save_image('Landmark visitation counts', itr)
        plt.close()

        # 9. Landmarks by spatial location
        landmarks_grid = env.visited.T.copy()
        landmarks_grid[landmarks_grid == 0] = -1
        landmarks_grid[landmarks_grid != -1] = 0
        
        node_labels = defaultdict(list)
        for i, position in enumerate(map(tuple, self.agent.landmarks.positions)):
            node_labels[position[1], position[0]].append(i)
            landmarks_grid[position[1], position[0]] += 1

        figure = plt.figure(figsize=(7, 7))
        plt.imshow(landmarks_grid)
        for pos, nodes in node_labels.items():
            plt.text(pos[1] -0.25, pos[0] + 0.25, ','.join(map(str, nodes)), fontsize=6)
        circle = plt.Circle(tuple(env.start_pos), 0.2, color='r')
        plt.gca().add_artist(circle)
        circle = plt.Circle(tuple(env.true_goal_pos), 0.2, color='purple')
        plt.gca().add_artist(circle)
        plt.colorbar()
        save_image('Landmarks', itr)
        plt.close()

        # 10. Landmarks graph
        #       - black edges have had successful transitions between their incident nodes
        #       - red edges were used to connect the graph and do not have any successful transitions
        G = self.agent.eval_landmarks.graph
        if self.agent.landmarks.num_landmarks <= 20:
            figure = plt.figure(figsize=(7, 7)) 
            pos = nx.circular_layout(G)

            non_zero_edges = {}
            zero_edges = {}
            for index, edge_info in dict(G.edges).items():
                if self.agent.eval_landmarks.zero_edge_indices is not None and index in self.agent.eval_landmarks.zero_edge_indices:
                    zero_edges[index] = edge_info
                else:
                    non_zero_edges[index] = edge_info
            
            nx.draw_networkx_nodes(G, pos, node_size=600)
            nx.draw_networkx_edges(G, pos, edgelist=non_zero_edges, width=2, edge_color='black', connectionstyle='arc3,rad=0.1')
            nx.draw_networkx_edges(G, pos, edgelist=zero_edges, width=2, edge_color='red', connectionstyle='arc3,rad=0.1')

            edge_labels = nx.get_edge_attributes(G, 'weight')
            for k, v in edge_labels.items():
                edge_labels[k] = round(v, 3)
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            nx.draw_networkx_edge_labels(G, pos, font_size=6, font_family='sans-serif', edge_labels=edge_labels)
            plt.axis('off')
            save_image('Landmarks graph', itr)
            plt.close()

        # # 11. Neighbors of the goal landmark
        # summary_writer = logger.get_tf_summary_writer()
        # goal_neighbors = ', '.join('({}, {:.3f})'.format(node, data['weight']) for node, data in G[0].items())
        # summary_writer.add_text("Goal neighbors", goal_neighbors, itr)

        # 12. Landmarks add threshold (dynamically adjusted)
        logger.record_tabular_stat('Add Threshold', self.agent.landmarks.add_threshold, itr)

        # 13. Landmarks low attempt threshold
        logger.record_tabular_stat('Low Attempt Threshold', self.agent.landmarks.get_low_attempt_threshold(use_max=False), itr)

        self.agent.reset_logging()

class MinibatchVizDoomLandmarkDSREval(MinibatchLandmarkDSREval):
    _eval = True

    def __init__(self,
                 figsize=(7, 7),
                 **kwargs):
        save__init__args(locals())
        super().__init__(**kwargs)
    
    # def eval(self, train_dir, eval_itrs=None):
    #     self.startup()
    #     device = self.agent.device 
    #     files = sorted(list(os.listdir(train_dir)))
    #     for f in files:
    #         match = re.search('itr_([0-9]+).pkl', f)
    #         if match:
    #             itr = match.group(1)
    #             if eval_itrs is None or itr is in eval_itrs:
    #                 landmarks_checkpoint = 'landmarks_itr_{}.npz'.format(itr)
    #                 if landmarks_checkpoint in files:
    #                     with logger.prefix(f"itr #{itr}"):
    #                         self.pbar = ProgBarCounter(1)
    #                         checkpoint = os.path.join(train_dir, f)
    #                         agent_state_dict = torch.load(checkpoint, map_location=device)['agent_state_dict']
    #                         feature_model_checkpoint = agent_state_dict['feature_model']
    #                         model_checkpoint = agent_state_dict['model']

    #                         self.agent.feature_model(feature_model_checkpoint)
    #                         self.agent.model.load_state_dict(model_checkpoint)
    #                         self.agent.target_model.load_state_dict(model_checkpoint)
                            
    #                         landmarks_checkpoint = os.path.join(train_dir, landmarks_checkpoint)
    #                         self.agent.landmarks.load(landmarks_checkpoint, device)

    #                         self.agent.landmarks.activate()
    #                         eval_traj_infos, eval_time = self.evaluate_agent(itr)
    #                         self.pbar.update(1)
    #                         self.log_diagnostics(itr, eval_traj_infos, eval_time)

    def _log_infos(self, traj_infos=None, itr=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for setting_name in self.sampler.eval_collector.eval_settings:
                for k in traj_infos[0][1]:
                    if not k.startswith("_"):
                        logger.record_tabular_misc_stat('{}{}'.format(setting_name, k),
                            [info[k] for name, info in traj_infos if name == setting_name], itr)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v, itr)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)        

    def evaluate_agent(self, itr):
        # Save first evaluation run to file
        eval_env = self.sampler.eval_collector.envs[0]
        eval_env.set_record_files([os.path.join(logger.get_snapshot_dir(), 'eval_run_0_itr_{}.lmp'.format(itr))])
        if itr > 0:
            self.pbar.stop()
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler.
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for name, info in eval_traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))
        self._cum_eval_time += eval_time
        logger.record_tabular_stat('CumEvalTime', self._cum_eval_time, itr)
        MinibatchRlBase.log_diagnostics(self, itr, eval_traj_infos, eval_time)
        env = self.sampler.collector.envs[0]
        state_entropy = entropy(env.visited.flatten(), base=2)
        if not np.isnan(state_entropy):
            logger.record_tabular_stat('Entropy', state_entropy, itr)
        state_entropy = entropy(env.visited_interval.flatten(), base=2)
        if not np.isnan(state_entropy):
            logger.record_tabular_stat('Interval Entropy', state_entropy, itr)

    def log_feature_info(self, itr, feature_info):
        summary_writer = logger.get_tf_summary_writer()
        for key, value in feature_info.items():
            summary_writer.add_histogram(key, value, itr)

    def log_dsr(self, itr):
        summary_writer = logger.get_tf_summary_writer()

        self.agent.sample_mode(itr)

        # 1. Render actual environment
        env = self.sampler.collector.envs[0]
        if itr < (self.log_dsr_interval_itrs * 2):
            figure = plt.figure(figsize=self.figsize)
            env.plot_topdown()
            save_image('Environment', itr)
            plt.close()

            if env.sample_sectors:
                figure = plt.figure(figsize=(7, 7))
                env.plot_topdown()
                plt.scatter(env.sample_sectors[:, 0], env.sample_sectors[:, 1], c=env.sample_sectors[:, 2], s=100)
                plt.colorbar()
                save_image('Sectors', itr)
                plt.close()

        # 2. Heatmap of state vistations during training interval
        figure = plt.figure(figsize=self.figsize)
        plt.imshow(env.visited_interval.T, origin='lower')
        plt.colorbar()
        save_image('State Visitation Heatmap (Interval)', itr)
        plt.close()

        env.reset_logging()

        # 3. Heatmap of state vistations during training overall
        figure = plt.figure(figsize=self.figsize)
        plt.imshow(env.visited.T, origin='lower')
        plt.colorbar()
        save_image('State Visitation Heatmap (Overall)', itr)
        plt.close()

        # Retrieve feature and successor feature representations for all states
        features, s_features, target_s_features, positions = self.agent.get_representations(env)

        diff_s_features = torch.mean(s_features - target_s_features, dim=2)
        summary_writer.add_histogram('DiffSFeature', diff_s_features, itr)
        logger.record_tabular_stat('diffDSRNorm', torch.norm(diff_s_features, p=2, dim=0).mean().item(), itr)

        # torch.save(features, os.path.join(logger.get_snapshot_dir(), 'features_itr_{}.pt'.format(itr)))
        # torch.save(s_features, os.path.join(logger.get_snapshot_dir(), 'dsr_itr_{}.pt'.format(itr)))
        
        # Comparison index
        subgoal_index = 0

        percentiles = [90, 92.5, 95, 97.5]

        # 4. Distance visualization in feature space
        features_similarity = self.agent.get_representation_similarity(features, mean_axes=1, subgoal_index=subgoal_index)
        
        figure = plt.figure(figsize=self.figsize)
        env.plot_topdown(objects=False)
        plt.scatter(positions[:, 0], positions[:, 1], c=features_similarity)
        plt.scatter(*positions[subgoal_index, :2], label='Subgoal', marker='D', c='red')
        plt.colorbar()
        plt.legend()
        save_image('Cosine Similarity in Feature Space', itr)
        plt.close()

        similarity_thresholds = np.percentile(features_similarity, percentiles)

        # 5. Points with High Feature Similarity
        figure = plt.figure(figsize=self.figsize)
        env.plot_topdown(objects=False)
        for threshold in similarity_thresholds:
            plt.scatter(positions[features_similarity > threshold, 0], positions[features_similarity > threshold, 1],
                        label='Similarity > {:.3f}'.format(threshold))
        plt.scatter(*positions[subgoal_index, :2], label='Subgoal', marker='D')

        plt.legend()
        save_image('Points with High Feature Similarity', itr)
        plt.close()

        # # 5. T-SNE visualization of features
        # tsne_embeddings = self.agent.get_tsne(features, mean_axes=1)

        # figure = plt.figure(figsize=(7, 7))
        # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=env.sample_sectors[:, 2])
        # plt.colorbar()
        # save_image('T-SNE of Features', itr)
        # plt.close()

        # 6. Distance visualization in SF space
        s_features_similarity = self.agent.get_representation_similarity(s_features, mean_axes=(1, 2), subgoal_index=subgoal_index)
        q_values = self.agent.get_q_values(s_features, mean_axes=1, subgoal_index=subgoal_index)

        figure = plt.figure(figsize=self.figsize)
        env.plot_topdown(objects=False)
        plt.scatter(positions[:, 0], positions[:, 1], c=s_features_similarity)
        plt.scatter(*positions[subgoal_index, :2], label='Subgoal', marker='D', c='red')

        # for i, position in enumerate(positions):
        #     x, y = position
        #     action = q_values[i].argmax()
        #     plt.text(x - 1, y - 1, str(action), fontsize=6)

        plt.colorbar()
        plt.legend()
        save_image('Cosine Similarity in SF Space', itr)
        plt.close()

        similarity_thresholds = np.percentile(s_features_similarity, percentiles)

        # 7. Points with High SF Similarity
        figure = plt.figure(figsize=self.figsize)
        env.plot_topdown(objects=False)
        for threshold in similarity_thresholds:
            plt.scatter(positions[s_features_similarity > threshold, 0], positions[s_features_similarity > threshold, 1],
                        label='Similarity > {:.3f}'.format(threshold))
        plt.scatter(*positions[subgoal_index, :2], label='Subgoal', marker='D')

        plt.legend()
        save_image('Points with High SF Similarity', itr)
        plt.close()

        # # 8. T-SNE visualization of successor features
        # tsne_embeddings = self.agent.get_tsne(features, mean_axes=1)

        # figure = plt.figure(figsize=(7, 7))
        # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=env.sample_sectors[:, 2])
        # plt.colorbar()
        # save_image('T-SNE of SF', itr)
        # plt.close()

    def log_eval_landmarks(self, itr):
        # Log eval landmarks information
        if self.agent.landmarks and self.agent.reached_goal:
            summary_writer = logger.get_tf_summary_writer()
            logger.record_tabular_stat('EndDistanceToGoal', np.average(self.agent.landmarks.eval_distances), itr)

            eval_env_index = 0
            eval_env = self.sampler.eval_collector.envs[eval_env_index]

            if self.agent.landmarks.found_eval_path:
                eval_path_str = '\n'.join(','.join(map(str, path)) + ' ({:.3f})'.format(self.agent.landmarks.path_p[i]) for i, path in enumerate(self.agent.landmarks.possible_paths))
                summary_writer.add_text("Path to goal", eval_path_str, itr)

                # Path to goal
                figure = plt.figure(figsize=self.figsize)
                eval_env.reset()
                eval_env.plot_topdown()
                best_path = np.argmax(self.agent.landmarks.path_p)
                for landmark in self.agent.landmarks.possible_paths[best_path]:
                    pos = self.agent.landmarks.positions[landmark]
                    plt.text(pos[0] - 0.25, pos[1] + 0.25, str(landmark), fontsize=10)
                landmark_positions = self.agent.landmarks.positions[self.agent.landmarks.possible_paths[best_path]]
                plt.scatter(landmark_positions[:, 0], landmark_positions[:, 1])
                save_image('Eval path to goal', itr)
                plt.close()

                # Agent's end position after executing landmark mode
                figure = plt.figure(figsize=self.figsize)
                eval_env.reset()
                eval_env.plot_topdown()
                eval_trajectory = self.sampler.eval_collector.eval_trajectories[eval_env_index]
                plt.scatter(eval_trajectory[:, 0], eval_trajectory[:, 1], c=range(len(eval_trajectory)))
                for pos, landmark in self.agent.landmarks.eval_end_pos.items():
                    plt.text(pos[0] - 0.25, pos[1] + 0.25, str(landmark), fontsize=10)
                    plt.scatter(pos[0], pos[1], marker='d', c='red')
                plt.colorbar()
                save_image('Eval end positions', itr)
                plt.close()

            # # State visitation heatmap in evaluation mode
            # figure = plt.figure(figsize=(7, 7))
            # plt.imshow(eval_env.visited_interval.T, origin='lower')
            # plt.colorbar()
            # save_image('Eval visitations', itr)
            # plt.close()

            eval_env.reset_logging()

    @torch.no_grad()
    def log_landmarks(self, itr):
        # If no landmark info to log
        if not self.agent.landmarks or self.agent.landmarks.num_landmarks == 0:
            return
        
        self.agent.sample_mode(itr)

        env = self.sampler.collector.envs[0]

        # Save landmarks data
        self.agent.landmarks.save(os.path.join(logger.get_snapshot_dir(), 'landmarks_itr_{}.npz'.format(itr)))

        # 1. Statistics related to adding/removing landmarks
        logger.record_tabular_stat('LandmarksAdded', self.agent.landmarks.landmark_adds, itr)
        logger.record_tabular_stat('LandmarksRemoved', self.agent.landmarks.landmark_removes, itr)
        
        # 2. Statistics related to transitions between landmarks
        # overall_success_rates = self.agent.landmarks.successes / np.clip(self.agent.landmarks.attempts, 1, None)
        # interval_success_rates = self.agent.landmarks.interval_successes / np.clip(self.agent.landmarks.interval_attempts, 1, None)

        # oracle_overall_success_rates = overall_success_rates[(self.agent.landmarks.attempts > 0)]
        # oracle_interval_success_rates = interval_success_rates[(self.agent.landmarks.interval_attempts > 0)]
        # logger.record_tabular_stat('OverallLandmarkSuccessRate',
        #                            np.average(oracle_overall_success_rates), itr)
        # logger.record_tabular_stat('IntervalLandmarkSuccessRate',
        #                            np.average(oracle_interval_success_rates), itr)
        average_random_steps = self.agent.landmarks.edge_random_steps / np.clip(self.agent.landmarks.edge_random_transitions, 1, None)
        average_random_steps = average_random_steps[self.agent.landmarks.edge_random_transitions > 0]
        average_subgoal_steps = self.agent.landmarks.edge_subgoal_steps / np.clip(self.agent.landmarks.edge_subgoal_transitions, 1, None)
        average_subgoal_steps = average_subgoal_steps[self.agent.landmarks.edge_subgoal_transitions > 0]
        # average_subgoal_successes = self.agent.landmarks.edge_subgoal_successes / np.clip(self.agent.landmarks.edge_subgoal_transitions, 1, None)
        # average_subgoal_successes = average_subgoal_successes[self.agent.landmarks.edge_subgoal_transitions > 0]
        logger.record_tabular_stat('TransitionRandomSteps',
                                   np.average(average_random_steps), itr)
        logger.record_tabular_stat('TransitionSubgoalSteps',
                                   np.average(average_subgoal_steps), itr)
        # logger.record_tabular_stat('transitionsubgoalsuccesses',
        #                            np.average(average_subgoal_successes), itr)
        
        subgoal_failures = self.agent.landmarks.edge_subgoal_failures
        logger.record_tabular_stat('TransitionSubgoalFailures',
                                   np.average(subgoal_failures[subgoal_failures > 0]), itr)
        logger.record_tabular_stat('UniqueSubgoalFailures', np.sum(self.agent.landmarks.edge_subgoal_failures > 0), itr)

        # logger.record_tabular_stat('UniqueSubgoalSuccceses', np.sum(self.agent.landmarks.edge_subgoal_successes > 0), itr)

        # 3. Statistics related to connected components of graph
        logger.record_tabular_stat('GraphConnectedComponents', np.average(self.agent.landmarks.graph_components), itr)
        logger.record_tabular_stat('GraphSizeLargestComponent',
                                   np.average(self.agent.landmarks.graph_size_largest_component), itr)
        
        # 3. Positions of landmark successes
        # figure = plt.figure(figsize=self.figsize)
        # success_positions = np.zeros_like(env.visited)
        # binned_positions = self.agent.landmarks.positions[:, :2].copy()
        # binned_positions[:, 0] = np.round(binned_positions[:, 0] - env.min_x).astype(int) // env.bin_size
        # binned_positions[:, 1] = np.round(binned_positions[:, 1] - env.min_y).astype(int) // env.bin_size
        # np.add.at(success_positions, (binned_positions[:, 0], binned_positions[:, 1]), np.sum(self.agent.landmarks.edge_subgoal_successes, axis=0))
        # plt.imshow(success_positions.T, origin='lower')
        # plt.colorbar()
        # save_image('Landmark successes', itr)
        # plt.close()
        
        # 4. Visitation counts of landmarks
        figure = plt.figure(figsize=self.figsize)
        visitations = np.sum(self.agent.landmarks.edge_random_transitions, axis=0) + np.sum(self.agent.landmarks.edge_subgoal_transitions, axis=0)
        plt.bar(np.arange(len(visitations)), visitations)
        plt.xlabel('Landmark')
        plt.ylabel('Visitations')
        save_image('Landmark visitation counts', itr)
        plt.close()

        # 4. Landmarks by spatial location
        figure = plt.figure(figsize=self.figsize)
        env.plot_topdown()
        if self.agent.landmarks.num_landmarks <= 50:
            node_labels = defaultdict(list)
            for i, position in enumerate(map(tuple, self.agent.landmarks.positions)):
                node_labels[position[1], position[0]].append(i)

            for pos, nodes in node_labels.items():
                plt.text(pos[1] -0.25, pos[0] + 0.25, ','.join(map(str, nodes)), fontsize=10)
        else:
            plt.scatter(self.agent.landmarks.positions[:, 0], self.agent.landmarks.positions[:, 1], c=range(self.agent.landmarks.num_landmarks))
            plt.colorbar()
        save_image('Landmarks', itr)
        plt.close()

        # 5. Distance to algo-chosen start landmark
        if self.agent.landmarks.dist_start_landmark:
            logger.record_tabular_stat('EstimatedStartLandmarkDistance',
                                       np.average(self.agent.landmarks.dist_start_landmark), itr)

        # 6. Ratio of distance to correct start landmark over distance
        #    to estimated start landmark
        if self.agent.landmarks.dist_ratio_start_landmark:
            logger.record_tabular_stat('Correct-EstimatedStartLandmarkDistanceRatio',
                                       np.average(self.agent.landmarks.dist_ratio_start_landmark), itr)

        # 7. Distance / angle diff at algo-determined termination
        if self.agent.landmarks.dist_at_termination:
            logger.record_tabular_stat('DistanceToLandmarkAtTermination',
                                       np.average(self.agent.landmarks.dist_at_termination), itr)
            logger.record_tabular_stat('AngleDifferenceAtTermination',
                                       np.average(self.agent.landmarks.angle_diff_at_termination), itr)
            logger.record_tabular_stat('WallIntersectionsAtTermination',
                                       self.agent.landmarks.wall_intersections_at_termination, itr)
            logger.record_tabular_stat('CorrectTerminationRatio',
                                       self.agent.landmarks.correct_terminations / np.clip(self.agent.landmarks.attempted_terminations, 1, None), itr)

        # 8. Distance / angle diff at algo-determined localization
        if self.agent.landmarks.dist_at_localization:
            logger.record_tabular_stat('DistanceToLandmarkAtLocalization',
                                       np.average(self.agent.landmarks.dist_at_localization), itr)
            logger.record_tabular_stat('AngleDifferenceAtLocalization',
                                       np.average(self.agent.landmarks.angle_diff_at_localization), itr)
            logger.record_tabular_stat('WallIntersectionsAtLocalization',
                                       self.agent.landmarks.wall_intersections_at_localization, itr)
            logger.record_tabular_stat('CorrectLocalizationRatio',
                                       self.agent.landmarks.correct_localizations / np.clip(self.agent.landmarks.attempted_localizations, 1, None), itr)

        # 8. Landmarks graph
        #       - black edges have had successful transitions between their incident nodes
        #       - red edges were used to connect the graph and do not have any successful transitions
        G = self.agent.eval_landmarks.graph
        if self.agent.landmarks.num_landmarks <= 20:
            figure = plt.figure(figsize=(7, 7))
            pos = nx.circular_layout(G)

            non_zero_edges = {}
            zero_edges = {}
            for index, edge_info in dict(G.edges).items():
                if self.agent.eval_landmarks.zero_edge_indices is not None and index in self.agent.eval_landmarks.zero_edge_indices:
                    zero_edges[index] = edge_info
                else:
                    non_zero_edges[index] = edge_info
            
            nx.draw_networkx_nodes(G, pos, node_size=600)
            nx.draw_networkx_edges(G, pos, edgelist=non_zero_edges, width=2, edge_color='black', connectionstyle='arc3,rad=0.1')
            # nx.draw_networkx_edges(G, pos, edgelist=zero_edges, width=2, edge_color='red', connectionstyle='arc3,rad=0.1')

            edge_labels = nx.get_edge_attributes(G, 'weight')
            for k, v in edge_labels.items():
                edge_labels[k] = round(v, 3)
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            nx.draw_networkx_edge_labels(G, pos, font_size=6, font_family='sans-serif', edge_labels=edge_labels)
            plt.axis('off')
            save_image('Landmarks graph', itr)
            plt.close()

        # # 11. Neighbors of the goal landmark
        # summary_writer = logger.get_tf_summary_writer()
        # goal_neighbors = ', '.join('({}, {:.3f})'.format(node, data['weight']) for node, data in G[0].items())
        # summary_writer.add_text("Goal neighbors", goal_neighbors, itr)

        # 12. Landmarks add threshold (dynamically adjusted)
        logger.record_tabular_stat('Add Threshold', self.agent.landmarks.add_threshold, itr)

        # # 13. Landmarks low attempt threshold
        # logger.record_tabular_stat('Low Attempt Threshold', self.agent.landmarks.get_low_attempt_threshold(use_max=False), itr)

        # # 14. Landmarks edge auxiliary threshold
        # logger.record_tabular_stat('Graph Edge Threshold', self.agent.landmarks.current_edge_threshold, itr)

        # # 15. Landmarks edge SF sim threshold
        # logger.record_tabular_stat('Graph SF Sim Threshold', self.agent.landmarks.current_sim_threshold, itr)
        
        # # 16. Attempts used to generate fully connected landmark graph 
        # logger.record_tabular_stat('Generate Graph Attempts', np.average(self.agent.landmarks.generate_graph_attempts), itr)

        # 13. Localization metrics
        logger.record_tabular_stat('Localization Transitions', self.agent.landmarks.transitions, itr)

        self.agent.reset_logging()
