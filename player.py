"""
this file loads weights, plays 20 games, and reports avg reward (number of lines cleared)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from utils import eval_weights
import sys

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

filename = sys.argv[1]
outfile = sys.argv[2]
with open(filename) as f:
	weights = json.load(f)
weights = np.array(weights["best_weights"])
print("weights:", weights)

all_lines_cleared = eval_weights(num_episodes=20, weights=weights)

# print(f"On average, {all_lines_cleared.mean()} lines were cleared")
# Print statistics
print(all_lines_cleared)
print("Avg lines cleared:", all_lines_cleared.mean())
print("Max lines cleared:", all_lines_cleared.max())
print("95% confidence interval:", mean_confidence_interval(all_lines_cleared))

plt.hist(all_lines_cleared, bins=20)
plt.xlabel("Lines Cleared")
plt.ylabel("Counts")
plt.title(f"Lines Cleared across {len(all_lines_cleared)} players")
plt.savefig(outfile)
plt.show()
