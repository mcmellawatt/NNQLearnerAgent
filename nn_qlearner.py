import numpy as np
from sklearn.neural_network import MLPRegressor


class NNQLearnerAgent():
    def __init__(self, n_actions):
        self._neural_net = MLPRegressor(warm_start=True)
        self._pure_explore_act_count = 0
        self._pure_explore_act = 0
        self._reward_avg_100_eps = 0
        self._epsilon = 1.0
        self._ep_reward = 0.0
        self._ep_rewards = []
        self._exp_replay = []
        self._prev_obs = None
        self._last_action_idx = 0
        self._freeze_count = 0

        self._N_ACTIONS = n_actions
        self._ACTION_ENCODINGS = self._initialize_action_encodings()
        self._IS_NN_INITIALIZED = False
        self._PREV_STATE_INDEX = 0
        self._ACTION_INDEX = 1
        self._REWARD_INDEX = 2
        self._MAX_Q_INDEX = 3
        self._DONE_INDEX = 4
        self._PURE_EXPLORE_N = 25
        self._GAMMA = 0.999
        self._BATCH_SIZE = 50
        self._EXP_REPLAY_HISTORY_SIZE = 20000
        self._FREEZE_PERIOD = 2000
        self._FREEZE_TRAIN_AVG = 50
        self._FREEZING_DISABLED = False
        self._EPSILON_DECAY = 0.995
        self._EPSILON_MINIMUM = 0.05

        # Clear any saved rewards from the previous run
        open("rewards", 'w').close()

    def choose_action(self, obs, reward, done, episode):
        # Accumulate the reward
        self._ep_reward += reward

        # Handle the terminating state
        if done:
            self._handle_done()

        # Build the experience replay entry
        curr_obs = np.atleast_2d(obs)

        if self._prev_obs is None:
            self._prev_obs = curr_obs
            best_act_idx = 0
            self._last_action_idx = best_act_idx
            return best_act_idx

        max_q_value = self._get_max_q_value(curr_obs)

        replay_entry = 5 * [0]
        replay_entry[self._PREV_STATE_INDEX] = self._prev_obs.copy()
        replay_entry[self._ACTION_INDEX] = self._last_action_idx
        replay_entry[self._REWARD_INDEX] = reward
        replay_entry[self._MAX_Q_INDEX] = max_q_value
        replay_entry[self._DONE_INDEX] = done

        self._exp_replay.append(replay_entry)
        self._prev_obs = curr_obs.copy()

        while len(self._exp_replay) >= self._EXP_REPLAY_HISTORY_SIZE:
            self._exp_replay.pop(0)

        # Add some episodes of pure exploration to the experience replay history
        if episode <= self._PURE_EXPLORE_N:
            if self._pure_explore_act_count < 10:
                self._pure_explore_act_count += 1
            else:
                self._pure_explore_act = np.random.randint(0, self._N_ACTIONS)
                self._pure_explore_act_count = 0
            best_act_idx = self._pure_explore_act
        else:
            # Train if freezing is disabled or we are below the average at which we'd like to freeze
            if self._FREEZING_DISABLED or self._reward_avg_100_eps < self._FREEZE_TRAIN_AVG:
                self._train_nn()
            else:
                # Start freezing
                if self._freeze_count > self._FREEZE_PERIOD:
                    self._train_nn()
                    self._freeze_count = 0
                else:
                    self._freeze_count += 1

            # Epsilon greedy
            if np.random.random() > self._epsilon:
                best_act_idx = self._get_best_action_idx(self._prev_obs)
            else:
                best_act_idx = np.random.randint(0, self._N_ACTIONS)

        self._last_action_idx = best_act_idx
        return best_act_idx

    def _train_nn(self):
        # Get a random sample
        sample = np.copy(self._exp_replay)
        np.random.shuffle(sample)

        previous_observations, prev_actions, discounted_qs, max_q_values, done = [], [], [], [], []

        # Build collections from sample
        for i in range(self._BATCH_SIZE):
            previous_observations.append(sample[i][self._PREV_STATE_INDEX])
            prev_actions.append(sample[i][self._ACTION_INDEX])
            discounted_qs.append(sample[i][self._REWARD_INDEX])
            max_q_values.append(sample[i][self._MAX_Q_INDEX])
            done.append(sample[i][self._DONE_INDEX])

        # Discount Qs if not in a done state
        for i in range(len(max_q_values)):
            if done[i] is False:
                discounted_qs[i] += self._GAMMA * max_q_values[i]

        # Build neural net input
        prev_sa_pairs = []
        for n in range(len(previous_observations)):
            prev_sa_pairs.append(self._get_sa_pair(previous_observations[n], prev_actions[n]))

        # Train the neural net
        self._neural_net.fit(np.asarray(prev_sa_pairs), np.asarray(discounted_qs))
        self._IS_NN_INITIALIZED = True

    # Used to encode the action with the observation as a unit vector
    def _initialize_action_encodings(self):
        action_encodings = np.zeros((self._N_ACTIONS, self._N_ACTIONS))
        for n in range(self._N_ACTIONS):
            action_encodings[n][n] = 1
        return action_encodings

    # Gets the maximum Q value for the available actions using the neural net
    def _get_max_q_value(self, observation):
        q_value = 0
        if self._IS_NN_INITIALIZED:
            sa_pairs = []

            for a in self._ACTION_ENCODINGS:
                sa_pair = np.concatenate((observation[0], a))
                sa_pairs.append(sa_pair)

            q_values = self._neural_net.predict(np.asarray(sa_pairs))
            q_value = np.max(q_values)
        return q_value

    # Gets the best action using the neural net
    def _get_best_action_idx(self, obs):
        sa_pairs = []
        for a in self._ACTION_ENCODINGS:
            sa_pairs.append(np.concatenate((obs[0], a)))

        q_values = self._neural_net.predict(np.asarray(sa_pairs))
        return np.argmax(q_values)

    # Gets an entry for the neural net input
    def _get_sa_pair(self, observation, action_idx):
        return np.concatenate((observation[0], self._ACTION_ENCODINGS[action_idx]))

    # Decays the epsilon by a fixed amount
    def _decay_epsilon(self):
        decayed_eps = self._epsilon * self._EPSILON_DECAY
        self._epsilon = decayed_eps if decayed_eps > self._EPSILON_MINIMUM else self._EPSILON_MINIMUM

    # Handles the done state (prints, resets reward, etc)
    def _handle_done(self):
            self._ep_rewards.append(self._ep_reward)
            self._reward_avg_100_eps = np.mean(self._ep_rewards[-100:])
            print ('Reward', self._ep_reward, 'Average Reward', round(self._reward_avg_100_eps, 2))
            outfile = open("rewards", 'a')
            outfile.write(str(self._ep_reward) + "\n")
            outfile.close()
            self._ep_reward = 0.0
            self._decay_epsilon()

