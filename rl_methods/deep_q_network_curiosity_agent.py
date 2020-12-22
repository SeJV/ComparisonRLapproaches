from typing import Optional, List, Tuple
from gym import Env
import numpy as np
from rl_methods import DeepQNetworkAgent
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Nadam


class DeepQNetworkCuriosityAgent(DeepQNetworkAgent):
    """
    TODO: test behavior of intrinsic reward, does it go down over time?
    TODO: Store png of model graph into folder of the agent
    TODO: ensure with following the existence of the directory of the model
        from pathlib import Path
        Path("/my/directory").mkdir(parents=True, exist_ok=True)
    """
    def __init__(self, env: Env, epsilon_start: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha_start: float = 0.01, alpha_min: Optional[float] = None, gamma: float = 0.99,
                 train_size: int = 512, nn_shape: List[int] = (126, 126), memory_len: int = 500000,
                 auto_store_models: bool = False, icm_scale: float = 1, name='DeepQNetworkCuriosityAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, alpha_start=alpha_start,
                         alpha_min=alpha_min, gamma=gamma, train_size=train_size, nn_shape=nn_shape,
                         memory_len=memory_len, auto_store_models=auto_store_models, name=name)

        self.fe_size = 2
        self.icm_scale = icm_scale

        # with state and action, predicts next state shared with FE
        self.state_prediction_model, self.inverse_dynamics_model, self.fe = self._build_intrinsic_curiosity_models()

    def reset(self) -> None:
        super().reset()
        # with state and action, predicts next state shared with FE
        self.state_prediction_model, self.inverse_dynamics_model, self.fe = self._build_intrinsic_curiosity_models()

    def _build_intrinsic_curiosity_models(self) -> Tuple[Model, Model, Model]:
        fe = self._build_feature_encoding()

        state_input = Input(shape=self.state_space, name='state_input')
        next_state_input = Input(shape=self.state_space, name='next_state_input')

        fe_state_input = fe(state_input)
        fe_next_state_input = fe(next_state_input)
        action_input = Input(shape=(self.action_space, ), name='action_input')

        state_prediction = Concatenate()([fe_state_input, action_input])
        state_prediction = Dense(32, 'relu')(state_prediction)
        state_prediction = Dense(32, 'relu')(state_prediction)
        state_prediction = Dense(self.fe_size, 'relu')(state_prediction)

        inverse_dynamics = Concatenate()([fe_state_input, fe_next_state_input])
        inverse_dynamics = Dense(32, 'relu')(inverse_dynamics)
        inverse_dynamics = Dense(32, 'relu')(inverse_dynamics)
        inverse_dynamics = Dense(self.action_space, 'relu')(inverse_dynamics)

        self.state_prediction_model = Model(inputs=[state_input, action_input], outputs=[state_prediction])
        self.inverse_dynamics_model = Model(inputs=[state_input, next_state_input], outputs=[inverse_dynamics])

        self._compile_models()
        return self.state_prediction_model, self.inverse_dynamics_model, fe

    def _build_feature_encoding(self) -> Model:
        fe_input = Input(shape=self.state_space, name='feature_encoding')
        fe = Flatten()(fe_input)
        fe = Dense(32, 'relu')(fe)
        fe = Dense(self.fe_size, 'relu')(fe)
        return Model(inputs=[fe_input], outputs=[fe])

    def _replay(self) -> None:
        mem_batch_idx = np.random.randint(len(self.memory), size=self.train_size)
        mem_batch = np.array(self.memory)[mem_batch_idx]

        states = np.squeeze(np.stack(mem_batch[:, 0]))
        actions = np.squeeze(np.stack(mem_batch[:, 1]))
        extrinsic_rewards = mem_batch[:, 2]
        next_states = np.squeeze(np.stack(mem_batch[:, 3]))
        dones = mem_batch[:, 4].astype(bool)

        # include intrinsic reward
        # 1. use information to create intrinsic reward
        next_states_fe = self.fe.predict(next_states)
        next_state_predictions_fe = self.state_prediction_model.predict([states, actions])
        intrinsic_rewards = np.sum(np.square(next_state_predictions_fe - next_states_fe))

        rewards = intrinsic_rewards + extrinsic_rewards

        # 2. use information to train inverse dynamics model
        self.inverse_dynamics_model.fit([states, next_states], [actions], batch_size=32, verbose=0)

        # 3. use information to train state prediction model
        self.state_prediction_model.fit([states, actions], [next_states_fe], batch_size=32, verbose=0)

        # Q(s,a) ← Q(s,a) + α(reward + γ max(Q(s_next)) − Q(s,a))
        # Here, only computing td target: reward + γ max(Q(s_next))
        next_q_values = self.target_model.predict(next_states)
        next_q_values[dones] = np.zeros(self.action_space)
        estimate_optimal_future = np.max(next_q_values, axis=-1).flatten()
        td_target = rewards + self.gamma * estimate_optimal_future

        q_vals = self.q_model.predict(states).reshape((self.train_size, self.action_space))
        q_vals[np.arange(len(q_vals)), np.argmax(actions, axis=-1)] = td_target  # override q_vals where action was taken with td_target

        self.q_model.fit(states, q_vals, batch_size=32, verbose=0)

    def store_models(self) -> None:
        super().store_models()
        self.state_prediction_model.save(f'models/{self.name}/state_prediction')
        self.inverse_dynamics_model.save(f'models/{self.name}/inverse_dynamics')
        self.fe.save(f'models/{self.name}/fe')
        # TODO: store models with interconnections, idea: self.full_model with all interconnections to store

    def load_models(self) -> None:
        self.state_prediction_model = load_model(f'models/{self.name}/state_prediction')
        self.inverse_dynamics_model = load_model(f'models/{self.name}/inverse_dynamics')
        self.fe = load_model(f'models/{self.name}/fe')
        super().load_models()
        # TODO: load models with interconnections, idea: self.full_model deconstruct into the three models

    def _compile_models(self) -> None:
        super()._compile_models()
        self.state_prediction_model.compile(optimizer=Nadam(lr=self.alpha), loss='mse')
        self.inverse_dynamics_model.compile(optimizer=Nadam(lr=self.alpha), loss='mse')

