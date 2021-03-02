import numpy as np
from gym import Env
from agents.abstract_agent import AbstractAgent


class DPAgent(AbstractAgent):
    def __init__(self, env: Env, theta: float = 0.1, gamma: float = 0.99, name: str = 'DPAgent'):
        super().__init__(env, name=name)
        self.mdp = env.P  # where self.mdp[state][action] gives a list of (probability, state t+1, reward, done)
        self.theta = theta
        self.gamma = gamma

        self.epsilon, self.epsilon_start, self.epsilon_min = 0, 0, 0

        self.episode = 1

        # only discrete environments possible
        self.state_space: int = self.env.observation_space.n
        self.action_space: int = self.env.action_space.n

        self.v_table = np.zeros(self.state_space)  # np.random.rand(self.state_space), self.v_table[-1] = 0.0
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.policy = np.full((self.state_space, self.action_space), 1 / self.action_space)
        self.policy_discrete = np.argmax(self.policy, -1)

        self._iterative_policy()  # update q_table

    def act(self, observation: int) -> int:
        return self.policy_discrete[observation]

    def _iterative_policy(self) -> None:
        has_policy_changed = True

        while has_policy_changed:
            # Policy evaluation until v-values stable
            delta = self.theta
            while not delta < self.theta:
                delta = self._ip_evaluation()

            # update q_values
            self._update_q_table()

            # update policy
            self._ip_improvement()

            self.episode += 1

            has_policy_changed = self._has_policy_changed()

    def _ip_evaluation(self) -> float:
        delta = 0
        v_new = self.v_table.copy()
        for state in range(self.state_space):
            s = 0
            for action in range(self.action_space):
                for (next_state_p, next_state, r, _) in self.mdp[state][action]:
                    s += self.policy[state, action] * next_state_p * (r + self.gamma * self.v_table[next_state])
            v_new[state] = s
            delta = max(delta, abs(v_new[state] - self.v_table[state]))

        self.v_table = v_new
        return delta

    def _update_q_table(self) -> None:
        for state in range(self.state_space):
            for action in range(self.action_space):
                s = 0
                for (next_state_p, next_state, r, _) in self.mdp[state][action]:
                    s += next_state_p * (r + self.gamma * self.v_table[next_state])
                self.q_table[state, action] = s

    def _ip_improvement(self) -> None:
        for state in range(self.state_space):
            best_action = np.argmax(self.q_table[state])
            self.policy[state] = np.zeros(self.action_space)
            self.policy[state, best_action] = 1.0

    def _has_policy_changed(self) -> bool:
        old_policy = self.policy_discrete
        new_policy = np.argmax(self.policy, -1)

        self.policy_discrete = new_policy
        return not np.array_equal(old_policy, new_policy)


