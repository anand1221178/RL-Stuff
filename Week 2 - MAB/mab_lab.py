import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, k=10, reward_variance=1.0, mean_variance=3.0, rng=None):
        self.k = k
        self.reward_variance = reward_variance
        self.mean_variance = mean_variance
        self.rng = rng if rng is not None else np.random.default_rng()
        self.true_means = self.rng.normal(0.0, np.sqrt(self.mean_variance), k)

    def pull(self, action):
        mu = self.true_means[action]
        return self.rng.normal(mu, np.sqrt(self.reward_variance))


def epsilon_greedy(k, steps, epsilon, alpha=None, runs=100, seed=0):
    master_rng = np.random.default_rng(seed)
    avg_reward = np.zeros(steps)
    for _ in range(runs):
        rng = np.random.default_rng(master_rng.integers(0, 2**32))
        bandit = MultiArmedBandit(k=k, rng=rng)
        q_vals = np.zeros(k)
        counts = np.zeros(k)
        for t in range(steps):
            if rng.random() < epsilon:
                act = rng.integers(0, k)
            else:
                max_val = q_vals.max()
                cand = np.where(np.isclose(q_vals, max_val))[0]
                act = rng.choice(cand)
            rew = bandit.pull(act)
            avg_reward[t] += rew
            counts[act] += 1
            if alpha is None:
                q_vals[act] += (rew - q_vals[act]) / counts[act]
            else:
                q_vals[act] += alpha * (rew - q_vals[act])
    avg_reward /= runs
    return avg_reward


def optimistic_greedy(k, steps, initial_q, alpha, runs=100, seed=0):
    master_rng = np.random.default_rng(seed)
    avg = np.zeros(steps)
    for _ in range(runs):
        rng = np.random.default_rng(master_rng.integers(0, 2**32))
        bandit = MultiArmedBandit(k=k, rng=rng)
        q_vals = np.full(k, initial_q, dtype=float)
        counts = np.zeros(k)
        for t in range(steps):
            max_val = q_vals.max()
            cand = np.where(np.isclose(q_vals, max_val))[0]
            act = rng.choice(cand)
            rew = bandit.pull(act)
            avg[t] += rew
            counts[act] += 1
            q_vals[act] += alpha * (rew - q_vals[act])
    avg /= runs
    return avg


def ucb(k, steps, c, runs=100, seed=0):
    master_rng = np.random.default_rng(seed)
    avg = np.zeros(steps)
    for _ in range(runs):
        rng = np.random.default_rng(master_rng.integers(0, 2**32))
        bandit = MultiArmedBandit(k=k, rng=rng)
        q_vals = np.zeros(k)
        counts = np.zeros(k)
        for t in range(steps):
            if (counts == 0).any():
                zero = np.where(counts == 0)[0]
                act = rng.choice(zero)
            else:
                ucb_values = q_vals + c * np.sqrt(np.log(t + 1) / counts)
                max_val = ucb_values.max()
                cand = np.where(np.isclose(ucb_values, max_val))[0]
                act = rng.choice(cand)
            rew = bandit.pull(act)
            avg[t] += rew
            counts[act] += 1
            q_vals[act] += (rew - q_vals[act]) / counts[act]
    avg /= runs
    return avg


def run_lab():
    k = 10
    steps = 1000
    runs = 100
    #fixed hyperparameters
    eps_curve = epsilon_greedy(k, steps, epsilon=0.1, alpha=None, runs=runs)
    opt_curve = optimistic_greedy(k, steps, initial_q=5.0, alpha=0.1, runs=runs)
    ucb_curve = ucb(k, steps, c=2.0, runs=runs)
    plt.figure(figsize=(8, 5))
    plt.plot(eps_curve, label="epsilon-greedy (epsilon=0.1)")
    plt.plot(opt_curve, label="optimistic greedy (Q1=5, alpha=0.1)")
    plt.plot(ucb_curve, label="UCB (c=2)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Reward over time (100-run average)")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("reward_over_time.png")
    plt.close()

    #hyperparameter study
    eps_vals = [0.01, 0.05, 0.1, 0.2, 0.5]
    eps_perf = [epsilon_greedy(k, steps, epsilon=x, alpha=None, runs=runs).mean() for x in eps_vals]
    q_vals = [0.0, 2.5, 5.0, 10.0]
    opt_perf = [optimistic_greedy(k, steps, initial_q=q, alpha=0.1, runs=runs).mean() for q in q_vals]
    c_vals = [0.5, 1.0, 2.0, 5.0]
    ucb_perf = [ucb(k, steps, c=v, runs=runs).mean() for v in c_vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_vals, eps_perf, marker='o', label="epsilon-greedy")
    ax.plot(q_vals, opt_perf, marker='o', label="optimistic greedy")
    ax.plot(c_vals, ucb_perf, marker='o', label="UCB")
    ax.set_xscale('log')
    ax.set_xlabel("Hyperparameter value (log scale)")
    ax.set_ylabel("Average reward over first 1000 steps")
    ax.set_title("Hyperparameter study: Average Reward vs Parameter")
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("hyperparam_study.png")
    plt.close()


if __name__ == "__main__":
    run_lab()