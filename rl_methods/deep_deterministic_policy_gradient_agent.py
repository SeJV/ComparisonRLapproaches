from typing import Optional, Tuple, List
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from rl_methods import AbstractAgent


class ReplayBuffer:
    def __init__(self, buffer_size: int = 100000, train_size: int = 64, observation_space: int = None,
                 action_space: int = None):
        self.buffer_size = buffer_size
        self.train_size = train_size

        self.record_counter: int = 0

        self.state_buffer = np.zeros((self.buffer_size, observation_space))
        self.action_buffer = np.zeros((self.buffer_size, action_space))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, observation_space))
        self.dones_buffer = np.zeros((self.buffer_size, 1)).astype(bool)

    # Takes (s,a,r,s',done) observation tuple as input
    def record(self, obs_tuple: Tuple[np.ndarray, float, float, np.ndarray, bool]) -> None:
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.record_counter % self.buffer_size

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.dones_buffer[index] = obs_tuple[4]

        self.record_counter += 1

    def get_training_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get sampling range
        record_range = min(self.record_counter, self.buffer_size)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.train_size)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        done_batch = np.squeeze(self.dones_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DeepDeterministicPolicyGradientAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha: float = 0.01, alpha_min: Optional[float] = None, gamma: float = 0.99,
                 train_size: int = 512, actor_shape: List[int] = (256, 256),
                 critic_shape: dict = None,
                 buffer_size: int = 100000, auto_store_models: bool = False, name: str = 'DDPGAgent'):
        """
        This implementation follows closely https://arxiv.org/pdf/1509.02971.pdf and
        https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb#scrollTo=5XXLGa-86N8a

        Changes to the implementations where made, so that this agent is easy usable and comparable with other agents
        of this repo.

        :param env:
        :param epsilon:
        :param epsilon_min:
        :param alpha:
        :param alpha_min:
        :param gamma:
        :param train_size:
        :param actor_shape:
        :param buffer_size:
        :param auto_store_models:
        :param name:
        """
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, alpha=alpha,
                         alpha_min=alpha_min, name=name)
        self.gamma = gamma
        self.train_size = train_size
        self.actor_shape = actor_shape
        self.critic_shape: dict = critic_shape if critic_shape is not None else {
            'state_path': [16, 32], 'action_path': [32], 'conc_path': [64, 64]
        }
        self.buffer_size = buffer_size
        self.auto_store_models = auto_store_models
        self.a: Optional[float] = None
        self.s: Optional[np.ndarray] = None

        # only continuous 1D state_space possible
        self.state_space = self.env.observation_space.shape[0]
        # action space is a single continuous value
        self.action_space = 1
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        self.buffer = ReplayBuffer(buffer_size=buffer_size, train_size=self.train_size,
                                   observation_space=self.state_space, action_space=self.action_space)

        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        self.target_actor_model = self.build_actor_model()
        self.target_critic_model = self.build_critic_model()
        self.tau = 0.005

        # Learning rate for actor-critic models
        self.critic_optimizer = Nadam(self.alpha * 2)
        self.actor_optimizer = Nadam(self.alpha)

        self._compile_models()

        # noise for exploration
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    def reset(self) -> None:
        super().reset()

        self.buffer = ReplayBuffer(train_size=self.train_size, observation_space=self.state_space,
                                   action_space=self.action_space)

        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        self.target_actor_model = self.build_actor_model()
        self.target_critic_model = self.build_critic_model()
        self._compile_models()

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.s = observation
        observation = np.expand_dims(observation, axis=0)
        action = np.squeeze(self.actor_model(observation))

        # add noise to action for exploration, scaled by epsilon
        action = action + self.epsilon * self.ou_noise()

        # Clip action to legal bounds
        self.a = np.clip(action, self.lower_bound, self.upper_bound)
        return self.a

    def train(self, s_next: np.ndarray, reward: float, done: bool) -> None:
        # storing sars' in buffer
        self.buffer.record((self.s, self.a, reward, s_next, done))
        if self.buffer.record_counter > self.train_size and self.buffer.record_counter % self.train_size == 0:
            self._replay()
            self.update_target(self.target_actor_model.variables, self.actor_model.variables)
            self.update_target(self.target_critic_model.variables, self.critic_model.variables)
            if self.buffer.record_counter % (self.train_size * 10) == 0 and self.auto_store_models:
                self.store_models()

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def _replay(self):
        states, actions, rewards, next_states, dones = self.buffer.get_training_sample()

        target_next_actions = self.target_actor_model(next_states, training=True)
        target_next_q_values = np.array(self.target_critic_model([next_states, target_next_actions], training=True))
        target_next_q_values[dones] = np.zeros(self.action_space)
        estimated_q_values = rewards + self.gamma * target_next_q_values

        self.critic_model.fit([states, actions], estimated_q_values, verbose=False)

        with tf.GradientTape() as tape:
            predicted_actions = self.actor_model(states, training=True)
            critic_value = self.critic_model([states, predicted_actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def build_actor_model(self):
        # Initialize weights between -3e-3 and 3-e3, so first result won't be -1 or 1 and gradient is not zero
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inp = m = Input(self.state_space)
        for layer in self.actor_shape:
            m = Dense(layer, 'relu')(m)
        m = Dense(self.action_space, 'tanh', kernel_initializer=last_init)(m)

        out_shift = (self.upper_bound + self.lower_bound) / 2   # center between upper and lower bound
        out_scale = (self.upper_bound - self.lower_bound) / 2   # distance from center to bounds
        out = (m + out_shift) * out_scale
        model = Model(inputs=inp, outputs=out)
        return model

    def build_critic_model(self):
        state_out = state_input = Input(self.state_space)
        for layer in self.critic_shape['state_path']:
            state_out = Dense(layer, 'relu')(state_out)
            state_out = Dense(layer, 'relu')(state_out)

        action_out = action_input = Input(self.action_space)
        for layer in self.critic_shape['action_path']:
            action_out = Dense(layer, 'relu')(action_out)

        out = Concatenate()([state_out, action_out])
        for layer in self.critic_shape['conc_path']:
            out = Dense(layer, 'relu')(out)
        out = Dense(1)(out)

        model = Model(inputs=[state_input, action_input], outputs=out)
        return model

    def _compile_models(self) -> None:
        self.critic_model.compile(self.critic_optimizer, loss='mse')
        # self.actor_model.compile('Nadam', loss='mse') # TODO! how to implement -value loss?

    def store_models(self) -> None:
        self.target_actor_model.save(f'models/{self.name}/actor_model')
        self.target_critic_model.save(f'models/{self.name}/critic_model')

    def load_models(self) -> None:
        self.actor_model = load_model(f'models/{self.name}/actor_model')
        self.target_actor_model = load_model(f'models/{self.name}/actor_model')

        self.critic_model = load_model(f'models/{self.name}/critic_model')
        self.target_critic_model = load_model(f'models/{self.name}/critic_model')
        self._compile_models()

    def episode_done(self, epsilon_reduction: float = 0, alpha_reduction: float = 0) -> None:
        self.epsilon = max(self.epsilon - epsilon_reduction, self.epsilon_min)

        self.alpha = max(self.alpha - alpha_reduction, self.alpha_min)
        self.critic_optimizer = Nadam(self.alpha * 2)
        self.actor_optimizer = Nadam(self.alpha)

        self._compile_models()


