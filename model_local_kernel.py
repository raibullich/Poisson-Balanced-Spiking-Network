# imports
import numpy as np
import matplotlib.pyplot as plt

class LocalBalancedNetworkKernel:
    """
    A class representing a locally balanced neural network model.

    This model simulates the dynamics of a network of neurons with balanced excitatory and inhibitory connections, 
    and can only take 1-dimensional input. The model uses Euler's method for numerical integration and includes 
    mechanisms for spiking activity based on conditional intensity.

    The special sauce in this model is the kernel to convolve the spike train (to give spikes some shape). Here you
    can specify what kernel you want to use from a set of predefined functions, or alternatively, provide your own kernel.

    Attributes:
        N (int): Number of neurons in the network.
        tdel (int): Time delay for recurrent connections.
        alpha (float): Gain parameter for the conditional intensity function.
        F_max (float): Maximum firing rate.
        F_min (float): Minimum firing rate.
        lam (float): Quadratic cost term (1/tau).
        mu (float): Linear cost term
        dt (float): Time step for the simulation.
        D (ndarray): Weight matrix with balanced positive and negative weights.
        threshold (ndarray): Spike threshold for each neuron.
        kernel (ndarray): shape of the decay function (spike)
    
    Methods:
        __init__(self, params):
            Initializes the network with given parameters.

        run(self, input):
            Runs the network simulation on the given input and kernel and returns various outputs including spike trains, 
            filtered spike trains, voltages, estimates, thresholds, weights, spike indices, and error.

        add PLOTTING FUNCTIONS

    """

    def __init__(self, params):
        """
        Initializes the LocalBalancedNetwork with the given parameters.

        Args:
            params (dict): A dictionary containing the following keys:
                'N' (int): Number of neurons.
                'tdel' (int): Time delay for recurrent connections.
                'alpha' (float): Gain parameter for the conditional intensity function.
                'F_max' (float): Maximum firing rate.
                'F_min' (float): Minimum firing rate.
                'tau' (float): Quadratic cost term
                'mu' (float): Linear cost term
                'dt' (float): Time step for the simulation.
                'D_mean' (float): Mean of the weight matrix D.
                'D_noise' (float): Standard deviation of the noise added to the weight matrix D.
        """
        self.N = params['N']            # Number of neurons.
        self.tdel = params['tdel']      # Time delay for recurrent connections.
        self.alpha = params['alpha']    # Gain parameter for the conditional intensity function.
        self.F_max = params['F_max']    # Maximum firing rate.
        self.F_min = params['F_min']    # Minimum firing rate.
        self.lam = 1 / params['tau']    # Quadratic cost term
        self.mu = params['mu']          # Linear cost term (omitted due to no ping-pong effect)
        self.T = params['T']            # Time simulation (s)
        self.dt = params['dt']          # Timestep
        
        # Generate weights
        self.D = np.ones((1, self.N))
        self.D[:, :(int(self.N / 2))] = -1                                      # Balanced positive vs negative weights
        self.D = (self.D / np.linalg.norm(self.D, axis=0) * params['D_mean'] + 
                  np.random.randn(1, self.N) * params['D_noise'])               # Normalize and add noise
        
        # Threshold
        self.threshold = (self.mu / self.lam ** 2 + np.diag(self.D.T @ self.D)) / 2

        # Identity matrix
        self.I = np.identity(self.N)

    def run(self, input, kernel):
            """
            Runs the network simulation on the given input.

            Args:
                input (ndarray): Input signal of shape (nT, 1), where nT is the number of time steps.
            
            Returns:
                tuple: A tuple containing the following elements:
                    - s (ndarray): Spike trains of shape (N, nT + 1 + tdel).
                    - r (ndarray): Filtered spike trains of shape (N, nT + 1 + tdel).
                    - v (ndarray): Voltages of shape (N, nT + 1 + tdel).
                    - estimate (ndarray): Estimate of the input signal.
                    - threshold (ndarray): Spike threshold for each neuron.
                    - D (ndarray): Weight matrix.
                    - spike_idx_neurons (list): List of arrays containing spike indices for each neuron.
                    - error (ndarray): Mean squared error between the input and the estimate.

            """
            self.input = input

            nT, nd = input.shape  # Extract number of time bins and dimensions of the input
            # d_input = np.gradient(input[:, 0]) / self.dt  # Derivative of the input
            
            # Initialize variables
            v = np.zeros((self.N, nT + 1 + self.tdel))  # Voltage
            s = np.zeros((self.N, nT + 1 + self.tdel))  # Spike train
            r = np.zeros((self.N, nT + 1 + self.tdel))  # Filtered spike trains
            prob_spike = np.zeros((self.N, nT + self.tdel))  # Probability of spike

            r[:, 0] = np.linalg.pinv(self.D) @ input[0]
            v[:, 0] = self.D.T @ (input[0] - self.D @ r[:, 0])
            
            # Run network simulation (Euler method)
            for t in range(nT):
           
                # Update voltage
                W_self = (np.diag(self.D.T @ self.D) + (self.mu / self.lam ** 2)) * self.I
                W_recurrent = self.D.T @ self.D
                W_recurrent[np.diag_indices(self.N)] = 0 # Set self connections to 0
                
                v[:, t+1] = self.D.T@input[t] - (W_self@r[:,t] + W_recurrent@r[:,t-self.tdel])

                # Compute spike rate and probability of spike (in self.dt)
                conditional_intensity = (self.F_max / 
                                        (1 + self.F_max * np.exp(-self.alpha * (v[:, t + 1] - self.threshold))) + 
                                        self.F_min)  # Conditional intensity
                prob_spike[:, t] = 1 - np.exp(-conditional_intensity * self.dt)  # Probability of spike

                # Spiking
                rand = np.random.rand(1, self.N)
                spike = np.where(prob_spike[:, t] > rand[0, :])[0]  # Find neurons that spike
                if len(spike):
                    s[spike, t + 1] = 1 / self.dt
                    r[spike, t+1:] += kernel[:nT-t + self.tdel] # update of filtered spike train

            # Get estimate
            estimate = self.D @ r

            # Get spike index for all neurons
            spike_idx_neurons = []
            for n in range(len(s)):
                spike_idx_neurons.append(np.where(s[n] != 0)[0])

            # Compute mean squared error (MSE) between estimate and input
            error = (input - estimate.T[:nT]) ** 2

            # save all outputs
            self.s = s
            self.r = r
            self.v = v
            self.estimate = estimate
            self.spike_idx_neurons = spike_idx_neurons
            self.error = error
            
            return s, r, v, estimate, self.threshold, self.D, spike_idx_neurons, error

    # Function to generate decaying functions (kernelss)
    def generate_kernel(self, kernel='bimodal'):
        
        # init variable of len of the simulation timesteps
        x = np.arange(0, self.T + self.dt *self.tdel, self.dt)

        if kernel=='bimodal':
            kernel = np.sin(x)*np.exp(-x/3)*10 # multiply a sin wave by negative exp to smooth out.
            kernel = kernel + abs(min(kernel)) # to avoid negative values # not really necessary, it improves estimate though
            kernel = kernel/max(kernel) # set it to have max height =1 (maye fully irrelevant, lets try out)
            kernel = kernel*np.exp(-x/5)
            return kernel
             
        elif kernel=='exponential':
            kernel = np.exp(-x) # using this kernel yelds same result as original network
            kernel = kernel/max(kernel) # set it to have max height =1 (maye fully irrelevant, lets try out)
            return kernel
        
        elif kernel=='gaussian':
            sigma = 0.6
            mu = 1.
            kernel = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)
            kernel = kernel/max(kernel) # set it to have max height =1
            return kernel

    # plotting functions
    # plot target output vs the network estimmate
    def plot_target_vs_estimate(self):

      x = np.arange(0, self.T, self.dt)
      fig, ax = plt.subplots(1, figsize=(15,5))
      ax.plot(x, self.estimate.T[:-1], label='estimate')
      ax.plot(x, self.input, label='target')
      ax.set(xlabel='Time (seconds)', ylabel='Relative current')
      ax.set_title('Target output vs Network estimate', fontsize=15)

      ax.legend(loc='upper right')
      fig.tight_layout()
      plt.show()

    # Plot spikes through time for all neurons (raster plot)
    def plot_raster(self):
        fig = plt.figure(figsize = (15,5))
        axs = fig.add_subplot()

        x = np.arange(0, self.T, self.dt)
        for n in range(self.N): # params['N']
            if self.D[:, n] > 0:
                color = 'C0'
            else:
                color = 'tab:red'
            axs.eventplot(x[self.spike_idx_neurons[n]], lineoffset = n+1, linelength = 0.5, colors = color)

        axs.set(xlabel = 'Time simulation (seconds)', ylabel = 'N = {}'.format(self.N), yticks = [], ylim=([0,self.N+1]))
        axs.set_title('Raster plot during simulation', fontsize=17)
        labels = [plt.Line2D([0], [0], color='C0', label = '$+$ weight'), plt.Line2D([0], [0], color = 'C3', label = '$-$ weight')]
        fig.legend(handles = labels, loc = 'upper center', bbox_to_anchor = (0.8, 0.98), ncol=2)
        plt.show()