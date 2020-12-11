from rl_methods import QLearningAgent


class QLearningCuriosityAgent(QLearningAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01, icm_scale=10):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, gamma=gamma, alpha=alpha)
        self.icm_scale = icm_scale
        self.seen_states_transitions = dict()

    def reset(self):
        super().reset()
        self.seen_states_transitions = dict()

    def train(self, s_next, extrinsic_reward):
        reward = extrinsic_reward + self._get_intrinsic_reward(self.s, self.a, s_next)
        super().train(s_next, reward)

    def _get_intrinsic_reward(self, state, action, next_state):
        # counting state transition occurrences. Additional intrinsic curiosity, if probability changes
        if (state, action) not in self.seen_states_transitions:
            self.seen_states_transitions[(state, action)] = {next_state: 0}
        elif next_state not in self.seen_states_transitions[(state, action)]:
            self.seen_states_transitions[(state, action)][next_state] = 0

        old_state_trans = self.seen_states_transitions[(state, action)].copy()  # old occurrences get temp stored
        self.seen_states_transitions[(state, action)][next_state] += 1  # new occurrences in self object

        intrinsic_reward = self._compare_probabilities(self.seen_states_transitions[(state, action)], old_state_trans,
                                                       next_state)
        return self.icm_scale * intrinsic_reward

    @staticmethod
    def _compare_probabilities(new_state_transitions, old_state_transitions, next_state):
        new_p = new_state_transitions[next_state] / sum(new_state_transitions.values())
        old_p = 0
        if sum(old_state_transitions.values()) != 0:
            old_p = old_state_transitions[next_state] / sum(old_state_transitions.values())

        return abs(new_p - old_p)
