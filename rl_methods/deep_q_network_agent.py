from typing import Optional, List
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from collections import deque
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from rl_methods import AbstractAgent


class DeepQNetworkAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha: float = 0.01, alpha_min: Optional[float] = None, gamma: float = 0.99,
                 train_size: int = 512, nn_shape: List[int] = (126, 126), memory_len: int = 500000,
                 auto_store_models: bool = False, name: str = 'DeepQNetworkAgent'):
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, alpha=alpha,
                         alpha_min=alpha_min, name=name)
        self.gamma = gamma
        self.train_size = train_size
        self.nn_shape = nn_shape
        self.memory_len = memory_len
        self.is_state_discrete = False
        self.auto_store_models = auto_store_models

        # only discrete actions possible, continuous and discrete state_space possible
        if isinstance(self.env.observation_space, Discrete):
            self.state_space = (self.env.observation_space.n, )
            self.is_state_discrete = True
        elif isinstance(self.env.observation_space, Box):
            self.state_space = self.env.observation_space.shape
        self.action_space = self.env.action_space.n

        self.episode = 1
        self.q_model = self.build_model()
        self.target_model = self.build_model()
        self._compile_models()
        self.s = None
        self.a = None
        self.memory = deque(maxlen=self.memory_len)

    def reset(self) -> None:
        super().reset()
        self.episode = 1
        self.q_model = self.build_model()
        self.target_model = self.build_model()
        self._compile_models()
        self.s = None
        self.a = None
        self.memory = deque(maxlen=self.memory_len)

    def build_model(self) -> Model:
        inp = Input(self.state_space)
        m = Flatten()(inp)
        for layer in self.nn_shape:
            m = Dense(layer, 'relu')(m)
        m = Dense(self.action_space)(m)

        model = Model(inputs=inp, outputs=m)
        return model

    def fast_predict(self, inp: np.ndarray) -> np.ndarray:
        weights = []
        for layer in range(2, len(self.nn_shape) + 3):  # input and flatten layer will get ignored, action layer added
            weights.append(self.q_model.layers[layer].get_weights())

        res = inp.flatten()
        for w in weights[:-1]:
            res = np.matmul(w[0].T, res) + w[1]
            res = res * (res > 0)  # relu

        return np.matmul(weights[-1][0].T, res) + weights[-1][1]  # linear

    def act(self, observation: np.ndarray) -> int:
        if self.is_state_discrete:
            one_hot = to_categorical(observation, self.state_space[0])
        else:
            one_hot = observation

        self.s = np.expand_dims(one_hot, axis=0)

        if np.random.random() > self.epsilon:
            action = np.argmax(self.fast_predict(self.s))
        else:
            action = np.random.randint(self.action_space)
        self.a = to_categorical(action, self.action_space)
        return action

    def train(self, s_next: np.ndarray, reward: float, done: bool) -> None:
        if self.is_state_discrete:
            one_hot = to_categorical(s_next, self.state_space[0])
        else:
            one_hot = s_next

        s_next = np.expand_dims(one_hot, axis=0)

        self.memory.append((
            self.s, self.a, reward, s_next, done
        ))
        if len(self.memory) > self.train_size and self.episode % self.train_size == 0:
            self._replay()
            if self.episode % (self.train_size * 10) == 0:
                self.target_model.set_weights(self.q_model.get_weights())
                self.target_model.make_predict_function()
                if self.auto_store_models:
                    self.store_models()

        self.episode += 1

    def store_models(self) -> None:
        self.q_model.save(f'models/{self.name}/q_model')

    def load_models(self) -> None:
        self.q_model = load_model(f'models/{self.name}/q_model')
        self.target_model = load_model(f'models/{self.name}/q_model')
        self._compile_models()

    def _compile_models(self) -> None:
        self.q_model.compile(loss='mse', optimizer=Nadam(lr=self.alpha))
        self.target_model.compile(loss='mse', optimizer=Nadam(lr=self.alpha))

    def _replay(self) -> None:
        mem_batch_idx = np.random.randint(len(self.memory), size=self.train_size)
        mem_batch = np.array(self.memory)[mem_batch_idx]

        states = np.squeeze(np.stack(mem_batch[:, 0]))
        actions = np.squeeze(np.stack(mem_batch[:, 1]))
        rewards = mem_batch[:, 2]
        next_states = np.squeeze(np.stack(mem_batch[:, 3]))
        dones = mem_batch[:, 4].astype(bool)

        # Q(s,a) ← Q(s,a) + α(reward + γ max(Q(s_next)) − Q(s,a))
        # Here, only computing td target: reward + γ max(Q(s_next))
        next_q_values = self.target_model.predict(next_states)
        next_q_values[dones] = np.zeros(self.action_space)
        estimate_optimal_future = np.max(next_q_values, axis=-1).flatten()
        td_target = rewards + self.gamma * estimate_optimal_future

        q_vals = self.q_model.predict(states).reshape((self.train_size, self.action_space))
        q_vals[np.arange(len(q_vals)), np.argmax(actions, axis=-1)] = td_target   # override q_vals where action was taken with td_target

        self.q_model.fit(states, q_vals, batch_size=32, verbose=0)
