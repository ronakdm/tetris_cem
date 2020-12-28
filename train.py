import json
import numpy as np
import sys

from utils import eval_weights

from joblib import Parallel, delayed

###################
# HYPERPARAMETERS #
###################

weights_dim = 6
goal = 10000

start_num_players = 25
end_num_players = 20
start_num_episodes = 20
end_num_episodes = 25

init_scaling = 100
num_players = start_num_players
elite_percent = 0.2
num_elite = max(int(num_players * elite_percent), weights_dim)
num_episodes = start_num_episodes
regularize_iters = 50  # iteration to stop regularizing covariance matrix
interpolate_percent = 0.7
num_best = 4

iters = 50


#################
# TRAINING LOOP #
#################

# Initialization
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print("Loading weights from file", filename, flush=True)
    with open(filename) as f:
        weights_dist = json.load(f)
        mu = np.array(weights_dist["mu"])
        cov = np.array(weights_dist["cov"])
else:
    mu = np.zeros(weights_dim)
    cov = init_scaling * np.eye(weights_dim)


# Regularization for Phase II training. Comment for Phase I.
# cov += 5 * np.eye(weights_dim)

prev_mu = mu.copy()
prev_cov = cov.copy()

elite_weights = []  # num_best elements from last iters elite set
prev_best = []

max_lines = []
mean_lines = []
var_lines = []
avg_trace_cov = []

for it in range(iters):

    prev_mu = mu.copy()
    prev_cov = cov.copy()

    # Draw weights from the distribution.
    weights = np.random.multivariate_normal(mu, cov, num_players)

    if len(prev_best) != 0:
        weights = np.concatenate([prev_best, weights])
        num_players += num_best

    # Evaluate players.
    players_lines_cleared = np.zeros(num_players)

    def worker(p):
        all_lines_cleared = eval_weights(num_episodes, weights[p])
        avg_lines_cleared = np.round(np.mean(all_lines_cleared), decimals=2)
        print(avg_lines_cleared, end=" ", flush=True)
        return avg_lines_cleared

    players_lines_cleared = np.array(
        Parallel(n_jobs=-2)(delayed(worker)(p) for p in range(num_players))
    )

    # Select num_elite weights from the players that had most number of lines cleared.
    elite_rewards_and_weights = sorted(
        zip(players_lines_cleared, weights), key=lambda pair: pair[0]
    )[-num_elite:]
    elite_rewards = np.array([reward for reward, _ in elite_rewards_and_weights])
    elite_weights = np.array([weight for _, weight in elite_rewards_and_weights])
    prev_best = elite_weights[-num_best:]

    # Recompute Gaussian distribution maximum likelihood estimates.
    mu = np.mean(elite_weights, axis=0)
    cov = np.cov(elite_weights, rowvar=False)

    # Regularize.
    cov += max((regularize_iters - it) / 10, 0) * np.eye(weights_dim)

    # Interpolate.
    mu = interpolate_percent * mu + (1.0 - interpolate_percent) * prev_mu
    cov = interpolate_percent * cov + (1.0 - interpolate_percent) * prev_cov

    # Recompute num_players and num_episodes.
    elite_lines_cleared = np.mean(elite_rewards)

    percent_finished = min(elite_lines_cleared / goal, 1)
    num_players = start_num_players + int(
        percent_finished * (end_num_players - start_num_players)
    )
    num_episodes = start_num_episodes + int(
        percent_finished * (end_num_episodes - start_num_episodes)
    )

    # Print statistics from this training iteration.
    print("Iteration %d: -----------------------------------------------" % it)
    print("Avg change in mu per dimension:", np.linalg.norm(mu - prev_mu) / weights_dim)
    print("Trace cov / dimension:", np.trace(cov) / weights_dim)
    print("Condition number of cov:", np.linalg.cond(cov))
    print("Mean weights:", mu)
    print("Max lines cleared by players:", players_lines_cleared.max())
    print("Avg lines cleared by players:", players_lines_cleared.mean())
    print("Variance in lines cleared by players:", players_lines_cleared.var())
    print()

    max_lines.append(players_lines_cleared.max())
    mean_lines.append(players_lines_cleared.mean())
    var_lines.append(players_lines_cleared.var())
    avg_trace_cov.append(np.trace(cov) / weights_dim)

    # Checkpointing.

    filename = "weights_checkpoints/metrics.txt"
    metrics = {
        "max_lines": max_lines,
        "mean_lines": mean_lines,
        "var_lines": var_lines,
        "avg_trace_cov": avg_trace_cov,
    }
    with open(filename, "w") as out:
        json.dump(metrics, out)

    filename = "weights_checkpoints/weights_dist_" + str(it) + ".txt"
    weights_dist = {"mu": mu.tolist(), "cov": cov.tolist()}
    with open(filename, "w") as out:
        json.dump(weights_dist, out)

    filename = "weights_checkpoints/best_weights_" + str(it) + ".txt"
    best_weights = {"best_weights": elite_weights[-1].tolist()}
    with open(filename, "w") as out:
        json.dump(best_weights, out)
