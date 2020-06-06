from collections import defaultdict
import copy
import itertools

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn_extra.cluster import KMedoids
import torch
import torch.nn.functional as F

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.dsr_agent import AgentInfo
from rlpyt.agents.dqn.dsr.feature_dsr_agent import FeatureDSRAgent, IDFDSRAgent, TCFDSRAgent
from rlpyt.agents.dqn.dsr.landmarks import Landmarks
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args


class LandmarkAgent(FeatureDSRAgent):

    def __init__(
            self,
            landmark_update_interval=int(5e3),
            add_threshold=0.75,
            max_landmarks=8,
            edge_threshold=0,
            use_affinity=True,
            affinity_decay=0.9,
            landmark_paths=None, 
            landmark_mode_interval=100,
            steps_per_landmark=25,
            reach_threshold=0.95,
            use_sf=False,
            use_soft_q=False,
            true_distance=False,
            steps_for_true_reach=2,
            oracle=False,
            use_true_start=False,
            use_oracle_landmarks=False,
            use_oracle_eval_landmarks=False,
            **kwargs):
        save__init__args(locals())
        self.explore = True
        self.landmarks = None
        self.landmark_mode_steps = 0
        self.reset_logging()
        super().__init__(**kwargs)
    
    def initialize(self, env_spaces, share_memory=False,
        global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.soft_distribution = Categorical(dim=env_spaces.action.n)

    def reset_logging(self):
        # Percentage of times we select the correct start landmark
        self.correct_start_landmark = 0
        self.total_landmark_paths = 0

        # Distance to algo-chosen start landmark / correct start landmark
        self.dist_ratio_start_landmark = []

        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks + 1)
        self.landmark_reaches = np.zeros(self.max_landmarks + 1)
        self.landmark_true_reaches = np.zeros(self.max_landmarks + 1)

        # End / start distance to ith landmark
        self.landmark_dist_completed = [[] for _ in range(self.max_landmarks + 1)]

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

        if self.landmarks is not None:
            self.landmarks.reset_logging()

    def set_oracle_landmarks(self, env):
        # Set oracle landmarks hardcoded in environment
        landmarks = env.get_oracle_landmarks()
        self.oracle_landmarks = Landmarks(len(landmarks), self.add_threshold, self.landmark_paths,
                                          self.use_affinity, self.affinity_decay)
        for landmark_obs, landmark_pos in landmarks:
            observation = torchify_buffer(landmark_obs).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self.oracle_landmarks.force_add_landmark(observation, features, dsr, landmark_pos)
        
        self.oracle_max_landmarks = len(landmarks)
        self.reset_logging()

    def set_env_true_dist(self, env):
        # Set map of true distances between any two points in environment
        self.env_true_dist = env.get_true_distances()

    @torch.no_grad()
    def create_landmarks(self, goal):
        # Create Landmarks object
        if self.use_oracle_landmarks:
            self.landmarks = self.oracle_landmarks
            self.max_landmarks = self.oracle_max_landmarks
        else:
            goal_obs, goal_pos = goal
            self.landmarks = Landmarks(self.max_landmarks, self.add_threshold, self.landmark_paths,
                                       self.use_affinity, self.affinity_decay)
            observation = torchify_buffer(goal_obs).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self.landmarks.add_landmark(observation, features, dsr, goal_pos)

    @torch.no_grad()
    def update_landmarks(self, itr):
        # Update features and DSR of existing landmarks
        if self.landmarks.num_landmarks:
            if (itr + 1) % self.landmark_update_interval == 0:
                observation = self.landmarks.observations

                model_inputs = buffer_to(observation,
                    device=self.device)
                features = self.feature_model(model_inputs, mode='encode')
                self.landmarks.set_features(features)

                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')
                self.landmarks.set_dsr(dsr)

                self.landmarks.update_affinity()

                self.landmarks.prune_landmarks()
                self.explore = True

    def landmark_mode(self):
        # Enter landmark mode during training every landmark_mode_interval steps
        if self.landmarks is not None and self.landmarks.num_landmarks > 0 and self.explore and \
            self._mode != 'eval' and (self.landmark_mode_steps % self.landmark_mode_interval) == 0:

                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None
                self.last_landmark = None

                # Select goal landmark with probability given by inverse of visitation count
                inverse_visitations = 1. / np.clip(self.landmarks.visitations, 1, None)
                landmark_probabilities = inverse_visitations / inverse_visitations.sum()
                self.goal_landmark = np.random.choice(range(len(landmark_probabilities)), p=landmark_probabilities)

    def generate_graph(self):
        # Use true distance edges
        if self.true_distance:
            self.landmarks.generate_true_graph(self.env_true_dist, self.edge_threshold)

        # Use DSR similarity edges
        else:
            self.landmarks.generate_graph(self.edge_threshold, self._mode)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, position=None):
        # TODO: Hack, we are currently only using one environment right now
        if position:
            position = position[0]

        if self.landmarks is not None:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            # Add landmarks if in exploration mode during training
            if self.explore and self._mode != 'eval' and not self.use_oracle_landmarks:
                self.landmarks.add_landmark(observation, features, dsr, position)

            # Select current (subgoal) landmark
            if not self.explore:

                # Select start landmark based on DSR similarity to current state
                if self.current_landmark is None:
                    norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
                    landmark_similarity = torch.matmul(self.landmarks.norm_dsr, norm_dsr.T)
                    self.current_landmark = landmark_similarity.argmax().item()

                    cur_x, cur_y = position

                    # Find correct start landmark based on true distances
                    closest_landmark = None
                    min_dist = None
                    for i, pos in enumerate(self.landmarks.positions):
                        dist = self.env_true_dist[cur_x, cur_y, pos[0], pos[1]]
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            closest_landmark = i

                        if i == self.current_landmark:
                            chosen_landmark_dist = dist

                    # Log if selected landmark is correct
                    self.total_landmark_paths += 1
                    if self.current_landmark == closest_landmark:
                        self.correct_start_landmark += 1

                    # Log ratio of distance to selected landmark / correct landmark
                    if chosen_landmark_dist != 0:
                        self.dist_ratio_start_landmark.append(min_dist / chosen_landmark_dist)
                    else:
                        self.dist_ratio_start_landmark.append(1)

                    # HACK: Use correct start landmark
                    if self.use_true_start:
                        self.current_landmark = closest_landmark

                    # Generate landmark graph and path between start and goal landmarks
                    self.generate_graph()
                    self.path = self.landmarks.generate_path(self.current_landmark, self.goal_landmark, self._mode)
                    self.path_idx = 0

                    # Log that we attempted to reach each landmark in the path
                    if self._mode != 'eval':
                        self.landmark_attempts[:len(self.path)] += 1

                    # Starting distance to goal landmark
                    landmark_x, landmark_y = self.landmarks.positions[self.goal_landmark]
                    self.start_distance_to_goal = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                    if self.start_distance_to_goal == 0:
                        self.start_distance_to_goal = 1
                    
                    self.start_distance_to_landmark = self.start_distance_to_goal

                # Loop on landmarks in path until we find one we have not reached in our current state
                find_next_landmark = True
                while find_next_landmark:

                    # If we still have steps remaining to try to reach the current landmark
                    if self.landmark_steps < self.steps_per_landmark:

                        # Log if current landmark is reached based on similarity
                        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                        landmark_similarity = torch.matmul(self.landmarks.norm_dsr[self.current_landmark], norm_dsr.T)
                        if landmark_similarity > self.reach_threshold and self._mode != 'eval':
                            self.landmark_reaches[self.path_idx] += 1
                        
                        # HACK: Check if current landmark is reached based on true distance
                        cur_x, cur_y = position
                        landmark_x, landmark_y = self.landmarks.positions[self.current_landmark]
                        end_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

                        # If goal landmark, we must try to reach it perfectly
                        if self.current_landmark == self.goal_landmark:
                            reach_threshold = 0
                        
                        # Otherwise, try to reach the current landmark within a true distance threshold
                        else:
                            reach_threshold = self.steps_for_true_reach

                        # If we have reached the current landmark based on our criteria
                        if end_distance <= reach_threshold:

                            # Increment the current landmark's visitation count
                            self.landmarks.visitations[self.current_landmark] += 1

                            # In training, log that landmark is truly reached and end/start distance to landmark ratio 
                            if self._mode != 'eval':
                                # TODO: Bucket by starting distance instead
                                self.landmark_true_reaches[self.path_idx] += 1
                                self.landmark_dist_completed[self.path_idx].append(end_distance / self.start_distance_to_landmark)

                                # Update landmark transition success rates
                                if self.last_landmark:
                                    self.landmarks.successes[self.last_landmark, self.current_landmark] += 1
                                    self.landmarks.successes[self.current_landmark, self.last_landmark] += 1
                                    self.landmarks.attempts[self.last_landmark, self.current_landmark] += 1
                                    self.landmarks.attempts[self.current_landmark, self.last_landmark] += 1

                            # If current landmark is goal, exit landmark mode
                            if self.current_landmark == self.goal_landmark:
                                self.explore = True
                                
                                # In eval, log end position trying to reach goal and distance away from goal
                                if self._mode == 'eval':
                                    self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                    self.eval_distances.append(end_distance)
                                
                                # In train, log end/start distance to goal ratio
                                else:
                                    # TODO: Bucket by starting distance
                                    self.goal_landmark_dist_completed.append(end_distance / self.start_distance_to_goal)
                                find_next_landmark = False
                            
                            # Else, move to next landmark and set as new subgoal
                            else:
                                self.last_landmark = self.current_landmark
                                self.path_idx += 1
                                self.current_landmark = self.path[self.path_idx]
                                self.landmark_steps = 0
                                find_next_landmark = True

                                landmark_x, landmark_y = self.landmarks.positions[self.current_landmark]
                                self.start_distance_to_landmark = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                        
                        # Otherwise, we continue to use subgoal policy to try to reach current landmark
                        else:
                            find_next_landmark = False
                    
                    # We have run out of steps trying to reach current landmark
                    else:
                        cur_x, cur_y = position 
                        landmark_x, landmark_y = self.landmarks.positions[self.current_landmark]
                        end_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

                        # In training, log end/start distance to landmark ratio
                        if self._mode != 'eval':
                            # TODO: Bucket by starting distance instead
                            self.landmark_dist_completed[self.path_idx].append(end_distance / self.start_distance_to_landmark)

                            if self.last_landmark:
                                self.landmarks.attempts[self.last_landmark, self.current_landmark] += 1
                                self.landmarks.attempts[self.current_landmark, self.last_landmark] += 1

                        # If current landmark is goal, exit landmark mode
                        if self.current_landmark == self.goal_landmark:
                            self.explore = True

                            # In eval, log end position trying to reach goal and distance away from goal
                            if self._mode == 'eval':
                                self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                self.eval_distances.append(end_distance)

                            # In train, log end/start distance to goal ratio
                            else:
                                # TODO: Bucket by starting distance instead
                                self.goal_landmark_dist_completed.append(end_distance / self.start_distance_to_goal)
                            find_next_landmark = False
                        
                        # Else, move on to next landmark and set as new goal
                        else:
                            self.last_landmark = None
                            self.path_idx += 1
                            self.current_landmark = self.path[self.path_idx]
                            self.landmark_steps = 0
                            find_next_landmark = True

                            landmark_x, landmark_y = self.landmarks.positions[self.current_landmark]
                            self.start_distance_to_landmark = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

        # Exploration (random) policy
        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)

            # In training, increment step counter used to track when to enter landmark mode
            if self._mode != 'eval':
                self.landmark_mode_steps += 1

        # Oracle landmark mode (in training only)
        elif self.oracle and self._mode != 'eval':
            cur_x, cur_y = position 
            landmark_x, landmark_y = self.landmarks.positions[self.current_landmark]
            new_pos = [[cur_x + 1, cur_y], [cur_x, cur_y + 1], [cur_x - 1, cur_y], [cur_x, cur_y - 1]]
            min_dist = None
            act = None
            for a, pos in enumerate(new_pos):
                new_x, new_y = pos
                dist = self.env_true_dist[new_x, new_y, landmark_x, landmark_y]
                if dist != -1:
                    if min_dist is None or dist < min_dist:
                        act = a
                        min_dist = dist

            action = torch.zeros_like(prev_action) + act
            self.landmark_steps += 1
        
        # Q landmark mode
        else:
            norm_dsr = dsr / torch.norm(dsr, p=2, dim=2, keepdim=True)

            # Use SF-based subgoal Q 
            if self.use_sf:
                subgoal_landmark_dsr = self.landmarks.norm_dsr[self.current_landmark]
                q_values = torch.matmul(norm_dsr, subgoal_landmark_dsr).cpu()
            
            # Use feature-based subgoal Q
            else:
                subgoal_landmark_features = self.landmarks.norm_features[self.current_landmark]
                q_values = torch.matmul(norm_dsr, subgoal_landmark_features).cpu()
            
            # Select action based on softmax of Q as probabilities
            if self.use_soft_q:
                prob = F.softmax(q_values, dim=1)
                action = self.soft_distribution.sample(DistInfo(prob=prob))
            
            # Select action based on epsilon-greedy of Q
            else:
                action = self.distribution.sample(q_values)
            self.landmark_steps += 1

        # Try to enter landmark mode
        self.landmark_mode()

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    def enter_eval_mode(self):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            # Using oracle landmarks, which includes goal as landmark
            if self.use_oracle_landmarks:
                pass
            
            # Using oracle landmarks in eval only, need to store training landmarks
            elif self.use_oracle_eval_landmarks:
                self.landmarks_storage = self.landmarks
                self.landmarks = self.oracle_landmarks

            self.first_reset = True
            self.eval_end_pos = defaultdict(list)
            self.eval_distances = []
            return True
        else:
            return False

    def reset(self):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            # In eval, always use landmarks mode
            if self._mode == 'eval':
                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None
                self.goal_landmark = 0
            
            # In training, start in exploration mode
            else:
                self.landmark_mode_steps = 0
                self.explore = True

    def reset_one(self, idx):
        self.reset()

    def exit_eval_mode(self):
        # Clean up evaluation mode
        if self.landmarks and self.landmarks.num_landmarks > 0:
            if self.use_oracle_landmarks:
                pass
            elif self.use_oracle_eval_landmarks:
                self.landmarks = self.landmarks_storage

            self.eval_goal = False

            # TODO: Re-continue landmark mode in training that was interrupted by evaluation
            self.explore = True

class LandmarkIDFAgent(LandmarkAgent, IDFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        IDFDSRAgent.__init__(self)

class LandmarkTCFAgent(LandmarkAgent, TCFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        TCFDSRAgent.__init__(self)