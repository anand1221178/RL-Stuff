import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k =10):
        self.k = k
        self.true_means = np.random.normal(0,3,k)
        self.std_dev = 1.0

    def pull(self, action):
        return np.random.normal(self.true_means[action], self.std_dev)
    
class EGreedyAgent:
    def __init__(self, k =10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k) #estimated values
        self.N = np.zeros(k) ##Counts of pulls per arm

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)
        
    def update(self, action, reward):
        self.N[action] += 1
        alpha = 1 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])

def run_bandit_exp(bandit, agent, steps=1000):
    rewards = np.zeros(steps)
    for t in range(steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards[t] = reward
    return rewards

def average_runs(epsilon=0.1, runs=100, steps=1000):
    all_rewards = np.zeros(steps)
    for _ in range(runs):
        bandit = Bandit(k=10)
        agent = EGreedyAgent(k=10, epsilon=epsilon)
        rewards = run_bandit_exp(bandit, agent, steps)
        all_rewards += rewards
    return all_rewards / runs

epsilons = [0.0, 0.1]
steps = 1000

for eps in epsilons:
    avg_rewards = average_runs(epsilon=eps, runs=100, steps=steps)
    plt.plot(avg_rewards, label=f"ε = {eps}")

plt.title("ε-Greedy: Average Reward over Time")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.show()

