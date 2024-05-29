
# imports
import numpy as np

# Network implementation of local balanced network from Rullán-Buxó and Pillow (2019)
# Update from their model: quadratic cost term is only now applied to the neuron's self-connection. Delays only in recurrent connections.

def run_local_framework(input, dt, params):
  '''
    Simulates a neural network with local balanced connections.

    Parameters:
    input (ndarray): Input signal that the network has to copy. Shape (nT, nd), where nT is the number of time bins and nd is the number of dimensions.
    dt (float): Size of the time steps for the simulation.
    params (dict): Dictionary containing the following parameters:
        - N (int): Number of neurons in the network.
        - tdel (int): Time delay in time bins for the recurrent connections.
        - alpha (float): Precision parameter for the conditional intensity function.
        - F_max (float): Maximum firing rate of the neurons.
        - F_min (float): Minimum firing rate of the neurons.
        - tau (float): Time constant for the decay rate of the filtered spike train.
        - mu (float): Quadratic cost term applied to the neuron's self-connection.
        - D_mean (float): Desired mean for the decoding weights.
        - D_noise (float): Standard deviation of the random noise added to the decoding weights.

    Returns:
    tuple: A tuple containing the following elements:
        - s (ndarray): Spike train of the neurons. Shape (N, nT + 1 + tdel).
        - r (ndarray): Filtered spike train. Shape (N, nT + 1 + tdel).
        - v (ndarray): Membrane potentials (voltage) of the neurons. Shape (N, nT + 1 + tdel).
        - estimate (ndarray): Estimated output signal of the network. Shape (nd, nT).
        - threshold (ndarray): Threshold values for the neurons. Shape (N,).
        - D (ndarray): Decoding weights. Shape (1, N).
        - spike_idx_neurons (list of ndarray): Indices of spikes for each neuron.
        - error (ndarray): Mean squared error between the input signal and the estimated output signal. Shape (nT, nd).
    '''
  
  ''' input: input = input that the network has to copu, dt = size of time steps of the simulation, params = parameters
      output: r = filtered spike train, s = spike train, v = voltage, estimate, threshold, D = decoding weights, spike_idx_neurons = spike indices, error = MSE'''
  
  # input
  input = input
  d_input = np.gradient(input[:,0])/dt # derivative of the input

  # extract parameters
  nT, nd = np.shape(input) # extract number of time bins and dimensions of the input
  N = params['N'] # number of neurons
  tdel = params['tdel'] # time delay in time bins

  # params for conditional intensity (nonlinearity)
  alpha = params['alpha'] # precision (change that)
  F_max = params['F_max'] # maximum firing rate
  F_min = params['F_min'] # minimum firing rate

  # dynamics
  lam = 1/params['tau'] # decay rate for filtered spike train

  # cost terms
  mu = params['mu'] # quadratic cost term
  spiking_cost = params['mu']*lam**2 # cost on spiking

  # generate weights
  D = np.ones((1,N))
  D[:,:(int(N/2))] = -1 # get balanced positive vs negative weights
  D =(D/np.linalg.norm(D, axis = 0)*params['D_mean'] + np.random.randn(1,N)*params['D_noise']) # normalize weights, set desired mean and add random noise
  print(D)

  #threshold
  threshold = (mu/lam**2 + np.diag(D.T@D))/2

  # initialize network variables
  v = np.zeros([N,nT+1+tdel]) # voltage
  s = np.zeros([N,nT+1+tdel]) # spike train
  r = np.zeros([N,nT+1+tdel]) # filtered spike trains
  prob_spike = np.zeros([N, nT+tdel]) #prob spike
  I = np.identity(N) # identity matrix (used to apply cost term only to the diagonal of the weight matrices, meaning that the cost term is only for one neuron!! just self connections)

  r[:,0] =np.linalg.pinv(D) @ input[0]
  v[:,0] = D.T @ (input[0]-D@r[:,0])

  # run network simulation (euler method)
  for t in range(nT):

    # update filtered spike trains
    dr = -lam*r[:,t] + s[:,t]
    r[:,t+1] = r[:,t] + dr*dt

    # update voltage
    W_self = (np.diag(D.T@D) + (mu/lam**2))*I # make sure of what is the ORDER of the matrix multiplication || self connections
    W_recurrent = D.T@D
    W_recurrent[np.diag_indices(N)] = 0 # set self connetions to 0 || these are recurrent connections
    dv = -lam*v[:,t] + D.T@(lam*input[t]+d_input[t]) - (W_self@s[:,t] + W_recurrent@s[:,t-tdel]) # only add a delay to the recurrent connection (NOT self)
    v[:,t+1] = v[:,t] + dv*dt

    # compute spike rate and probability of spike (in dt)
    conditional_intensity = F_max/(1+F_max*np.exp(-alpha*(v[:,t+1]-threshold))) + F_min # find contitional intensity (lambda): instantaneous firing rate
    prob_spike[:, t] = 1-np.exp(-conditional_intensity*dt) # convert firing rate to probability of spike

    # spiking
    rand = np.random.rand(1,N)
    spike = np.where(prob_spike[:, t] > rand[0,:])[0] # find neurons that spike
    # find neurons that spike
    if len(spike):
      s[spike, t+1] = 1/dt


  # get estimate
  estimate = D@r

  # get spike index for all neurons
  spike_idx_neurons = []
  for n in range(len(s)):
    spike_idx_neurons.append(np.where(s[n] != 0)[0])

  # compute mean squared error (MSE) between estimate vs input
  error = (input - estimate.T[:nT])**2

  return s, r, v, estimate, threshold, D, spike_idx_neurons, error