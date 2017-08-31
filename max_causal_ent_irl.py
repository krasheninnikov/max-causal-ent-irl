import numpy as np 
from frozen_lake import FrozenLakeEnv
from mdps import MDP, MDPOneTimeR
from traj_tools import generate_trajectories, compute_s_a_visitations
from value_iter_and_policy import vi_boltzmann, vi_rational 
from occupancy_measure import compute_D

def max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma=1, h=None, 
                       temperature=1, epochs=1, learning_rate=0.2, theta=None):
    '''
    Finds theta, a reward parametrization vector (r[s] = features[s]'.*theta) 
    that maximizes the log likelihood of the given expert trajectories, 
    modelling the expert as a Boltzmann rational agent with given temperature. 
    
    This is equivalent to finding a reward parametrization vector giving rise 
    to a reward vector giving rise to Boltzmann rational policy whose expected 
    feature count matches the average feature count of the given expert 
    trajectories (Levine et al, supplement to the GPIRL paper).

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    feature_matrix : 2D numpy array
        Each of the rows of the feature matrix is a vector of features of the 
        corresponding state of the MDP. 
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, state and action].
    gamma : float 
        Discount factor; 0<=gamma<=1.
    h : int
        Horizon for the finite horizon version of value iteration.
    temperature : float >= 0
        The temperature parameter for computing V, Q and policy of the 
        Boltzmann rational agent: p(a|s) is proportional to exp(Q/temperature);
        the closer temperature is to 0 the more rational the agent is.
    epochs : int
        Number of iterations gradient descent will run.
    learning_rate : float
        Learning rate for gradient descent.
    theta : 1D numpy array
        Initial reward function parameters vector with the length equal to the 
        #features.
    Returns
    -------
    1D numpy array
        Reward function parameters computed with Maximum Causal Entropy 
        algorithm from the expert trajectories.
    '''    
    
    # Compute the state-action visitation counts and the probability 
    # of a trajectory starting in state s from the expert trajectories.
    sa_visit_count, P_0 = compute_s_a_visitations(mdp, gamma, trajectories)
    
    # Mean state visitation count of expert trajectories
    # mean_s_visit_count[s] = ( \sum_{i,t} 1_{traj_s_{i,t} = s}) / num_traj
    mean_s_visit_count = np.sum(sa_visit_count,1) / trajectories.shape[0]
    # Mean feature count of expert trajectories
    mean_f_count = np.dot(feature_matrix.T, mean_s_visit_count)
    
    if theta is None:
        theta = np.random.rand(feature_matrix.shape[1])
        

    for i in range(epochs):
        r = np.dot(feature_matrix, theta)
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        V, Q, policy = vi_boltzmann(mdp, gamma, r, h, temperature)
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = np.sum(sa_visit_count * (Q - V))
        
        # The expected #times policy π visits state s in a given #timesteps.
        D = compute_D(mdp, gamma, policy, P_0, t_max=trajectories.shape[1])        

        # IRL log likelihood gradient w.r.t rewardparameters. 
        # Corresponds to line 9 of Algorithm 2 from the MaxCausalEnt IRL paper 
        # www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf. 
        # Negate to get the gradient of neg log likelihood, 
        # which is then minimized with GD.
        dL_dtheta = -(mean_f_count - np.dot(feature_matrix.T, D))

        # Gradient descent
        theta = theta - learning_rate * dL_dtheta

        if (i+1)%10==0: 
            print('Epoch: {} log likelihood of all traj: {}'.format(i,L), 
                  ', average per traj step: {}'.format(
                  L/(trajectories.shape[0] * trajectories.shape[1])))
    return theta


def main(t_expert=1e-2,
         t_irl=1e-2,
         gamma=1,
         h=10,
         n_traj=200,
         traj_len=10,
         learning_rate=0.01,
         epochs=300):
    '''
    Demonstrates the usage of the implemented MaxCausalEnt IRL algorithm. 
    
    First a number of expert trajectories is generated using the true reward 
    giving rise to the Boltzmann rational expert policy with temperature t_exp. 
    
    Hereafter the max_causal_ent_irl() function is used to find a reward vector
    that maximizes the log likelihood of the generated expert trajectories, 
    modelling the expert as a Boltzmann rational agent with temperature t_irl.
    
    Parameters
    ----------
    t_expert : float >= 0
        The temperature parameter for computing V, Q and policy of the 
        Boltzmann rational expert: p(a|s) is proportional to exp(Q/t_expert);
        the closer temperature is to 0 the more rational the expert is.
    t_irl : float
        Temperature of the Boltzmann rational policy the IRL algorithm assumes
        the expert followed when generating the trajectories.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    h : int
        Horizon for the finite horizon version of value iteration subroutine of
        MaxCausalEnt IRL algorithm.
    n_traj : int
        Number of expert trajectories generated.
    traj_len : int
        Number of timesteps in each of the expert trajectories.
    learning_rate : float
        Learning rate for gradient descent in the MaxCausalEnt IRL algorithm.
    epochs : int
        Number of gradient descent steps in the MaxCausalEnt IRL algorithm.
    '''
    np.random.seed(0)
    mdp = MDPOneTimeR(FrozenLakeEnv(is_slippery=False))    

    # Features
    feature_matrix = np.eye(mdp.nS)
    # Add dummy feature to show that features work
    if False:
        feature_matrix = np.concatenate((feature_matrix, np.ones((mdp.nS,1))), 
                                        axis=1)
    
    # The true reward weights and the reward
    theta_expert = np.zeros(feature_matrix.shape[1])
    theta_expert[24] = 1
    r_expert = np.dot(feature_matrix, theta_expert)
    
    # Compute the Boltzmann rational expert policy from the given true reward.
    if t_expert>0:
        V, Q, policy_expert = vi_boltzmann(mdp, gamma, r_expert, h, t_expert)
    if t_expert==0:
        V, Q, policy_expert = vi_rational(mdp, gamma, r_expert, h)
        
    # Generate expert trajectories using the given expert policy.
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    
    # Compute and print the stats of the generated expert trajectories.
    sa_visit_count, _ = compute_s_a_visitations(mdp, gamma, trajectories)
    log_likelihood = np.sum(sa_visit_count * (Q - V))
    print('Generated {} traj of length {}'.format(n_traj, traj_len))
    print('Log likelihood of all traj under the policy generated ', 
          'from the true reward: {}, \n average per traj step: {}'.format(
           log_likelihood, log_likelihood / (n_traj * traj_len)))
    print('Average return per expert trajectory: {} \n'.format(
            np.sum(np.sum(sa_visit_count, axis=1)*r_expert) / n_traj))

    # Find a reward vector that maximizes the log likelihood of the generated 
    # expert trajectories.
    theta = max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma, h, 
                               t_irl, epochs, learning_rate)
    print('Final reward weights: ', theta)

if __name__ == "__main__":
    main()