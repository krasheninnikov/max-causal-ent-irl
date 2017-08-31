import numpy as np


def generate_trajectories(mdp, policy, timesteps=20, num_traj=50):
    '''
    Generates trajectories in the MDP given a policy.
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    policy : 2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    timesteps : int
        Length of each of the generated trajectories.
    num_traj : 
        Number of trajectories to generate.
    
    Returns
    -------
    3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, 2: state & action].
    '''
    
    trajectories = np.zeros([num_traj, timesteps, 2]).astype(int)
    
    s = mdp.reset()
    for i in range(num_traj):
        for t in range(timesteps):
            action = np.random.choice(range(mdp.nA), p=policy[s, :])
            trajectories[i, t, :] = [s, action]
            s = mdp.step(action)
        s = mdp.reset()
    
    return trajectories


def compute_s_a_visitations(mdp, gamma, trajectories):
    '''
    Given a list of trajectories in an mdp, computes the state-action 
    visitation counts and the probability of a trajectory starting in state s.
    
    State-action visitation counts:
    sa_visit_count[s,a] = \sum_{i,t} 1_{traj_s_{i,t} = s AND traj_a_{i,t} = a}

    P_0(s) -- probability that the trajectory will start in state s. 
    P_0[s] = \sum_{i,t} 1_{t = 0 AND traj_s_{i,t} = s}  / i
    P_0 is used in computing the occupancy measure of the MDP.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, 2: state & action].

    Returns
    -------
    (2D numpy array, 1D numpy array)
        Arrays of shape (mdp.nS, mdp.nA) and (mdp.nS).
    '''

    s_0_count = np.zeros(mdp.nS)
    sa_visit_count = np.zeros((mdp.nS, mdp.nA))
    
    for traj in trajectories:
        # traj[0][0] is the state of the first timestep of the trajectory.
        s_0_count[traj[0][0]] += 1
        for (s, a) in traj:
            sa_visit_count[s, a] += 1
      
    # Count into probability        
    P_0 = s_0_count / trajectories.shape[0]
    
    return sa_visit_count, P_0