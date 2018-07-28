# Maximum Causal Entropy Inverse Reinforcement Learning

Python implementation of Maximum Causal Entropy Inverse Reinforcement Learning from Brian Ziebart's PhD thesis (2010): http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf

Main file: ```max_causal_ent_irl.py```.

Dependencies: ```numpy```, ```gym```.

## Algorithm notes

The finite horizon version of the algorithm is consistent and works as it should by Ziebart (2010).

The discounted infinite horizon version used in e.g Levine's Gaussian Process IRL is work in progress. Current TODOs:
1. Compute P_0 for the occupancy measure for the gradient of the infinite horizon discounted version of the algorithm as in section A of Levine's GPIRL supplement http://graphics.stanford.edu/projects/gpirl/gpirl_supplement.pdf.
