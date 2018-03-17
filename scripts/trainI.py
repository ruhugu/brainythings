# Find how the potential of a neuron changes when the external current is a train pulse of increasing amplitude.

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from brainythings import *

# Parameters
I_value_duration = 100  # Time the current spends at a certain value
relaxtime = 50  # Wait time between different I values as a fraction of I_dur
I_values = np.linspace(0, 10, 5)  # Values that the current takes 
figylen = 3.5
figxlen = 1.8*figylen

cycletime = relaxtime + I_value_duration
ts = np.linspace(0, cycletime*I_values.size, 10000)  # Measure times in


# Create the external current function
def I_ext(t, I_values, I_value_duration):
    """External current function.

    """
    cycletime = relaxtime + I_value_duration

    I_idx = 0
    intervallim = (I_idx + 1.)*cycletime
    while (t > intervallim) and (I_idx < I_values.size):
        I_idx += 1
        intervallim = (I_idx + 1.)*cycletime
    if (I_idx < I_values.size) and (intervallim - t < I_value_duration):
        I = I_values[I_idx]
    else:
        I = 0.
    return I

# Create the neuron and solve
neuron = HHNeuron()
neuron.I_ext = lambda t: I_ext(t, I_values, I_value_duration)
neuron.solve(ts=ts)

# Calculate the current at each time for the plot
Is = np.array([neuron.I_ext(t) for t in neuron.ts])
 
# Plot the results
fig = plt.figure(figsize=(figxlen, figylen))
ax_V = fig.add_subplot(211)
ax_I = fig.add_subplot(212, sharex=ax_V)
ax_V.set_ylabel("V {0}".format(neuron.V_unit))
ax_I.set_xlabel("Time {0}".format(neuron.time_unit))
ax_I.set_ylabel("I {0}".format(neuron.I_unit))

ax_V.plot(neuron.ts, neuron.Vs)
ax_I.plot(neuron.ts, Is)

fig.tight_layout()

# Show the plot
plt.show()

# Save the figure to file (this does not work when plt.show() has been used)
#fig.savefig("Itrain{0}-{1}.png".format(I_values[0], I_values[-1]), dpi=150)
