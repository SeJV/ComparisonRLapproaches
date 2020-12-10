from rl_methods import QLearningAgent


class QTableSimpleCuriosityAgent(QLearningAgent):
    def __init__(self, env, gamma=0.99, start_epsilon=1.0, epsilon_min=0.0, alpha=0.01, icm_scale=0.5):
        super().__init__(env, gamma, start_epsilon, epsilon_min, alpha)
        self.icm_scale = icm_scale
        self.seen_states_transitions = dict()

    def reset(self):
        super().reset()
        self.seen_states_transitions = dict()

    def train(self, s_next, extrinsic_reward):
        reward = extrinsic_reward + self._get_intrinsic_reward(self.s, self.a, s_next)
        super().train(s_next, reward)

    def _get_intrinsic_reward(self, state, action, next_state):
        observed_state = self.seen_states_transitions[(state, action)]
        if observed_state is None:
            observed_state = {next_state: 1}
        elif observed_state[next_state] is None:
            observed_state[next_state] = 1
        else:
            observed_state[next_state] += 1

        intrinsic_reward = self._compare_probabilities(observed_state, self.seen_states_transitions[(state, action)],
                                                       next_state)
        return self.icm_scale * intrinsic_reward

    @staticmethod
    def _compare_probabilities(new_state_transitions, old_state_transitions, next_state):
        new_p = new_state_transitions[next_state] / sum(new_state_transitions.values())
        old_p = old_state_transitions[next_state] / sum(old_state_transitions.values())

        return abs(new_p - old_p)
