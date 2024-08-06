# Poisson-Balanced-Spiking-Network

This is repositroy contains an implementation in python and some examples of the [Poisson Balanced Spiking Network](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008261), derived by Camille Rullán Buixó and Jonathan Pillow (2020). Their work was based on the framework of ['balanced networks'](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003258), first developed by Martin Boerlin and colleagues (2013). However, the Poisson component only came after from the hands of Camille Rullán Buixó and Jonathan Pillow, where their main contribution was to include a probabilistic threshold in the neurons of the network and connectivity delays. 

The implementation can be found in the file model_local.py, which is a very close implementation from the Poisson Balanced Spiking Network 'local' model (Rullán-Buixó & Pillow, 2020). Local refers to where the probabilistic term is included, which in this case is implemented at a neuron level. I have also included another implementation in a file called model_local_kernel. In this implementation the exponential decay function after a neuron spikes can be modified. This allows to try the network using different kernels, which could represent different spike shapes. Both models are brieffly explored in the notebook 'Simulation PBSNN.ipynb'.

This work is part of my master thesis supervised by Fleur Zeldernust and Sander Keemink, at the Donders Institute (Nijmegen, the Netherlands). If you want to know more details about the thesis and get an idea of what was the whole project about, feel free to take a look at it! You can find the thesis [here](https://theses.ubn.ru.nl/server/api/core/bitstreams/95edb5d2-6b0c-41c7-b615-a1ac4b502c5f/content).

### Probabilistic (soft) threshold and the activation function.

But what does it mean to have a probabilistic threshold? To be explained...

### Sketching the main idea and derivation of the network.

To be done...
