import numpy as np

class StimulusGenerator:
    """
    A class for generating various types of stimuli that can be used as input for the Poisson BSN network.

    Attributes:
        dt (float): Time step for the simulation.
        T (float): Total time for the simulation.

    Methods:
        stimulus_sin(): Generates a sinusoidal stimulus.
        stimulus_step_random(num_steps): Generates a random step stimulus.
        stimulus_step(num_steps: Generates a step stimulus with equal jumps.
        stimulus_convolved_random(): Generates a convolved random walk stimulus.
        stimulus_constant(y_value): Generates a constant stimulus.
    """

    def __init__(self, params):
        """
        Initializes the StimulusGenerator with the given parameters.

        Args:
            params (dict): A dictionary containing the following keys:
                'dt' (float): Time step for the simulation.
                'T' (float): Total time for the simulation.
        """
        self.dt = params['dt']
        self.T = params['T']
        self.input_range = [-20,20]

    def stimulus_sin(self):
        stimulus = np.array([self.input_range[1] * np.sin(np.pi * np.arange(0, self.T, self.dt))]).T
        # stimulus = np.vstack([self.input_range[1] * np.sin(2 * np.pi * np.arange(0, time, dt)), self.input_range[1] * np.cos(2 * np.pi * np.arange(0, time, dt))]).T
        return stimulus

    def stimulus_abs_sin(self):
        zero_input = np.zeros(int(self.T / (4 * self.dt)))  # zero input to stabilize the net
        sin_input = abs(np.sin(np.arange(0, 3 * self.T / 4, self.dt)))  # actual (sin) input
        stimulus = np.append(zero_input, sin_input)  # only positive input
        stimulus = stimulus * np.diff(self.input_range) + self.input_range[0]  # shift input to have desired range
        stimulus = np.reshape(stimulus, (len(stimulus), 1))  # reshape to have dimension information
        return stimulus

    def stimulus_step_random(self, num_steps):
        """num_steps is the number of jumps you want in your input"""
        time_steps = int(self.T / self.dt)  # number of time bins
        step_size = int(time_steps / num_steps)  # size of each step in time bins
        stimulus = np.zeros(time_steps)
        for i in range(num_steps):
            rand = np.random.rand()  # generate random step move
            stimulus[i * step_size:(i + 1) * step_size] = np.ones(step_size) * rand  # generate stimulus
        stimulus = stimulus * np.diff(self.input_range) + self.input_range[0]  # shift to have desired range
        stimulus[0:step_size] = 0  # set first block with 0
        stimulus = np.reshape(stimulus, (len(stimulus), 1))  # reshape to have dimension information
        return stimulus

    def stimulus_step(self, num_steps):
        time_steps = int(self.T / self.dt)  # number of time bins
        step_size = int(time_steps / num_steps)  # size of each step in time bins
        jump = np.diff(self.input_range) / num_steps  # make all jumps be the same size
        stimulus = np.zeros(time_steps)
        for i in range(num_steps):
            stimulus[i * step_size:(i + 1) * step_size] = np.zeros(step_size) + jump * i
        stimulus = stimulus / max(stimulus) * np.diff(self.input_range) + self.input_range[0]  # shift to have desired range
        stimulus = np.reshape(stimulus, (len(stimulus), 1))  # reshape to have dimension information
        return stimulus

    def stimulus_convolved_random(self):
        stimulus = np.zeros(int(self.T / self.dt))  # initialize
        for i in range(int(self.T / self.dt)):
            stimulus[i] = stimulus[i - 1] + np.random.normal() / 10
        ones = np.ones(1000) / 1000
        stimulus = np.convolve(stimulus, ones, mode='same')
        stimulus = (stimulus - min(stimulus)) / (max(stimulus) - min(stimulus))
        stimulus = stimulus * np.diff(self.input_range) + self.input_range[0]
        stimulus = stimulus - np.mean(stimulus)  # mean center to 0
        stimulus = np.reshape(stimulus, (len(stimulus), 1))  # reshape to have dimension information
        return stimulus

    def stimulus_constant(self, y_value):
        """y_value is where you want the constant input to be in the y axis"""
        stimulus = np.ones(int(self.T / self.dt) + 1) * y_value
        stimulus = np.reshape(stimulus, (len(stimulus), 1))
        return stimulus


