import numpy as np

class EpsilonGreedyBandit:
    """
    Each 'arm' is a candidate item to recommend.
    The bandit learns which arms get the most positive feedback (clicks/likes)
    and gradually favors them, while still exploring occasionally.
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = current_value + (reward - current_value) / n


def simulate_feedback_loop(bandit: EpsilonGreedyBandit, true_preferences: np.ndarray, rounds: int = 500):
    """
    true_preferences: ground-truth probability that the user likes each arm/item
    (used to SIMULATE feedback for the demo — in production this comes from real clicks)
    """
    history = []
    for _ in range(rounds):
        arm = bandit.select_arm()
        reward = np.random.binomial(1, true_preferences[arm])
        bandit.update(arm, reward)
        history.append((arm, reward))
    return history