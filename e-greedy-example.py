'''
Epsilon-Greedy Algorithm:
Simulation explore-exploit dilemma
using Multi-Armed Bandit analogy
'''
import matplotlib.pyplot as plt
import numpy as np

BANDIT_PROBABILITIES = [0.20,0.75,0.50] #real probabilities
TOTAL_TRIALS = 10000
EPSILON = 0.1
ALPHA = 0.99999

class Bandit:
    def __init__(self,win_rate):
        self.num_trials = 0
        self.win_rate_estimated = 0
        self.win_rate = win_rate

    def pull(self):
        rng = np.random.rand()
        value = rng < self.win_rate
        self.update(value)
        return value

    def update(self,value):
        reward_updated = self.win_rate_estimated * self.num_trials + value
        self.num_trials += 1
        self.win_rate_estimated = reward_updated / self.num_trials

def run_experiment(epsilon,decaying_epsilon = False):
    rewards = np.zeros(TOTAL_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_times_optimal_chosen = 0
    bandits = [Bandit(x) for x in BANDIT_PROBABILITIES]
    optimal_bandit = np.argmax([b.win_rate for b in bandits])

    ## Decaying epsilon parameters
    alpha = 1
    eps = epsilon

    for iteration in range(0,TOTAL_TRIALS):
        rng = np.random.rand()
        if rng < eps:
            num_times_explored += 1
            bandit_chosen = np.random.randint(0,len(bandits))
        else:
            num_times_exploited += 1
            bandit_chosen = np.argmax([b.win_rate_estimated for b in bandits])
        if bandit_chosen == optimal_bandit:
            num_times_optimal_chosen += 1
        rewards[iteration] = bandits[bandit_chosen].pull();
        if decaying_epsilon:
            alpha = alpha * ALPHA
            eps = eps * alpha

    print("\n-----EXPERIMENT RESULTS: (eps: %s)-----" % epsilon)
    print("Real win rate vs Estimated for each bandit:")
    for b in bandits:
        print(b.win_rate, "-",b.win_rate_estimated)

    print()
    print("Total reward earned: ",rewards.sum())
    print("Overall win rate: ",rewards.sum() / TOTAL_TRIALS)
    print("Times optimal chosen: ",num_times_optimal_chosen)
    print("Times explored: ",num_times_explored)
    print("Times exploited: ",num_times_exploited)
    print("Final epsilon: ", eps) ## useful in decaying epsilon option

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(TOTAL_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(TOTAL_TRIALS) * np.max(BANDIT_PROBABILITIES))
    # instead of linear(default) using log because the algorithm converges fast
    plt.xscale('log')
    plt.show()
    return win_rates

if __name__ == "__main__":
    # Comparing different epsilons
    # Higher epsilon (Quick Conversion)
    # Lower epsilon (Higuer Eventual Reward)
    e1 = run_experiment(0.1)
    e2 = run_experiment(0.05)
    e3 = run_experiment(0.01)

    # log
    plt.plot(e1, label= 'eps: 0.1')
    plt.plot(e2, label= 'eps: 0.05')
    plt.plot(e3, label= 'eps: 0.01')
    plt.plot(np.ones(TOTAL_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear
    plt.plot(e1, label= 'eps: 0.1')
    plt.plot(e2, label= 'eps: 0.05')
    plt.plot(e3, label= 'eps: 0.01')
    plt.plot(np.ones(TOTAL_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.legend()
    plt.show()
