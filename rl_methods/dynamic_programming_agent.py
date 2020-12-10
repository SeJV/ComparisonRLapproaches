import numpy as np


class DPAgent:
    def __init__(self, env, theta=0.01, gamma=1):
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n
        self.mdp = env.env.P  # where self.mdp[state][action] gives a list of (probability, state t+1, reward, done)
        self.theta = theta
        self.gamma = gamma

        self.episode = 1

        self.v_table = np.zeros(self.state_space)  # np.random.rand(self.state_space), self.v_table[-1] = 0.0
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.policy = np.full((self.state_space, self.action_space), 1 / self.action_space)
        self.policy_discrete = np.argmax(self.policy, -1)

    def choose_action(self, observation):
        return self.policy_discrete[observation]

    def iterative_policy(self):
        has_policy_changed = True

        while has_policy_changed:
            # Policy evaluation until v-values stable
            delta = self.theta
            while not delta < self.theta:
                delta = self.ip_evaluation()

            # update q_values
            self.update_q_table()

            # update policy
            self.ip_improvement()

            self.episode += 1

            has_policy_changed = self.has_policy_changed()

    def ip_evaluation(self):
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

    def update_q_table(self):
        for state in range(self.state_space):
            for action in range(self.action_space):
                s = 0
                for (next_state_p, next_state, r, _) in self.mdp[state][action]:
                    s += next_state_p * (r + self.gamma * self.v_table[next_state])
                self.q_table[state, action] = s

    def ip_improvement(self):
        for state in range(self.state_space):
            best_action = np.argmax(self.q_table[state])
            self.policy[state] = np.zeros(self.action_space)
            self.policy[state, best_action] = 1.0

    def has_policy_changed(self):
        old_policy = self.policy_discrete
        new_policy = np.argmax(self.policy, -1)

        self.policy_discrete = new_policy
        return not np.array_equal(old_policy, new_policy)


