from typing import Optional, Tuple, List
import numpy as np
from numpy.random import default_rng
from gym import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Nadam
from agents import AbstractAgent


class _ReplayBuffer:
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


class _OUActionNoise:
    def __init__(self, mean=0, std_deviation=1, theta=1, dt=1e-2, x_initial=None):
        """
        Random noise generator for single values
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = x_initial if x_initial is not None else 0
        self.reset()

    def __call__(self) -> float:
        rng = default_rng()
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * rng.normal()
        )
        # Store x into x_prev
        # Makes next noise dependent on previous
        self.x_prev = x
        return x

    def reset(self) -> None:
        self.x_prev = self.x_initial if self.x_initial is not None else 0


class DeepDeterministicPolicyGradientAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: float = 0,
                 epsilon_reduction: float = 0.0, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0.0, gamma: float = 0.99, train_size: int = 512, tau: float = 0.005,
                 actor_shape: List[int] = (64, 64), critic_shape: dict = None, buffer_size: int = 100000,
                 auto_store_models: bool = False, name: str = 'DDPGAgent'):
        """
        This implementation follows closely https://arxiv.org/pdf/1509.02971.pdf and
        https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb#scrollTo=5XXLGa-86N8a

        Changes to the implementations where made, so that this agent is easy usable and comparable with other agents
        of this repo.

        :param gamma: Discount of future rewards
        :param train_size: Sample size of training replay
        :param actor_shape: Shape of sequential dense actor NN
        :param critic_shape: Shape of dense critic NN. Has state and action path, which then will be concatenated
            resulting in dictionary like: {'state_path': [16, 32], 'action_path': [32], 'conc_path': [64, 64]}
        :param buffer_size: Size of replay buffer, where replay information is stored
        :param auto_store_models: if true, models of actor and critic will be stored after 10 replays
        """
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_reduction=epsilon_reduction,
                         alpha=alpha, alpha_min=alpha_min, alpha_reduction=alpha_reduction, name=name)
        self.gamma = gamma
        self.train_size = train_size
        self.actor_shape = actor_shape
        self.critic_shape: dict = critic_shape if critic_shape is not None else {
            'state_path': [32, 16], 'action_path': [32], 'conc_path': [32, 32]
        }
        self.buffer_size = buffer_size
        self.auto_store_models = auto_store_models
        self.a: Optional[float] = None
        self.s: Optional[np.ndarray] = None

        # only continuous 1D state_space possible
        self.state_space: int = self.env.observation_space.shape[0]
        # action space is a single continuous value
        self.action_space: int = 1
        self.upper_bound: float = self.env.action_space.high[0]
        self.lower_bound: float = self.env.action_space.low[0]

        self.buffer = _ReplayBuffer(buffer_size=buffer_size, train_size=self.train_size,
                                    observation_space=self.state_space, action_space=self.action_space)

        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        self.target_actor_model = self._build_actor_model()
        self.target_critic_model = self._build_critic_model()
        self.tau = tau

        # Learning rate for actor-critic models
        self.critic_optimizer = Nadam(self.alpha * 2)  # critic optimizer has twice the lr of actor optimizer
        self.actor_optimizer = Nadam(self.alpha)

        self._compile_models()

        # noise for exploration for single values
        self.ou_noise = _OUActionNoise()

    def reset(self) -> None:
        super().reset()

        self.buffer = _ReplayBuffer(train_size=self.train_size, observation_space=self.state_space,
                                    action_space=self.action_space)

        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        self.target_actor_model = self._build_actor_model()
        self.target_critic_model = self._build_critic_model()
        self._compile_models()

        self.ou_noise.reset()

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.s = observation
        observation = np.expand_dims(observation, axis=0)
        action = np.squeeze(self.actor_model(observation))

        # add noise to action for exploration, scaled by epsilon
        action = np.expand_dims(action + self.epsilon * self.ou_noise(), axis=0)
        # Clip action to legal bounds
        self.a = np.clip(action, self.lower_bound, self.upper_bound)
        return self.a

    def train(self, s_next: np.ndarray, reward: float, done: bool) -> None:
        # storing sars' in buffer
        self.buffer.record((self.s, self.a, reward, s_next, done))
        if self.buffer.record_counter > self.train_size and self.buffer.record_counter % self.train_size == 0:
            self._replay()
            self._update_target(self.target_actor_model.variables, self.actor_model.variables)
            self._update_target(self.target_critic_model.variables, self.critic_model.variables)
            if self.buffer.record_counter % (self.train_size * 10) == 0 and self.auto_store_models:
                self.store_models()

    def store_models(self) -> None:
        self.target_actor_model.save(f'models/{self.name}/actor_model')
        self.target_critic_model.save(f'models/{self.name}/critic_model')

    def load_models(self) -> None:
        self.actor_model = load_model(f'models/{self.name}/actor_model')
        self.target_actor_model = load_model(f'models/{self.name}/actor_model')

        self.critic_model = load_model(f'models/{self.name}/critic_model')
        self.target_critic_model = load_model(f'models/{self.name}/critic_model')
        self._compile_models()

    def episode_done(self) -> None:
        super().episode_done()

        self.critic_optimizer = Nadam(self.alpha * 2)
        self.actor_optimizer = Nadam(self.alpha)

        self._compile_models()

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def _update_target(self, target_weights, weights):
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
            critic_value = self.critic_model([states, predicted_actions], training=False)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_model.trainable_variables)
            )

    def _build_actor_model(self) -> Model:
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

    def _build_critic_model(self) -> Model:
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

