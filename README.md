# brainythings
 
Python implementation of several neuron models. Right now the implemented ones are the [Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) and  the [FitzHugh-Nagumo model](http://www.scholarpedia.org/article/FitzHugh-Nagumo_model).

## Required packages

This module requires [matplotlib](https://matplotlib.org/), [scipy](https://www.scipy.org/) and [numpy](http://www.numpy.org/) to be installed.


## Some output examples

Membrane potential of a Hodgkin-Huxley neuron under a train of current pulses with increasing amplitude ([script](https://github.com/ruhugu/brainythings/blob/master/scripts/trainI.py)).

<img src="https://raw.githubusercontent.com/ruhugu/brainythings/master/output_examples/Itrain0.0-10.0_HH.png" alt="Drawing" width="600"/>

Phase plane analysis of a FitzHugh-Nagumo neuron under a constant current I=0.85 (in the simulation units) ([script](https://github.com/ruhugu/brainythings/blob/master/scripts/phaseplane.py)).

<img src="https://raw.githubusercontent.com/ruhugu/brainythings/master/output_examples/phaseplaneI0.85_FN.gif" alt="Drawing" width="600"/>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
