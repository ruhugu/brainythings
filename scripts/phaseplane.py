# Phase plane analysis of the FitzHugh-Nagumo model.

import os
import sys
import numpy as np
from matplotlib import colors as colors
from matplotlib import pyplot as plt
from matplotlib import animation as animation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from brainythings import *

# Parameters
I_ampl = 0.324175
tmax = 100.
V_0 = -0.6
W_0 = -0.4

figylen = 3
figxlen = 2*figylen
fps = 20.
animspeed = 18.
traj_color = "C0"  # Color of trajectory in phase plane
plot_color = "C1"


# Create neuron and calculate trajectory
ts, t_step = np.linspace(0, tmax, 100*tmax, retstep=True)
neuron = FNNeuron(I_ampl=I_ampl, V_0=V_0, W_0=W_0)
neuron.solve(ts=ts)

# Range of V to be plotted
Vs_margin = 0.1*(np.amax(neuron.Vs) - np.amin(neuron.Vs))
Vs_range = [np.amin(neuron.Vs) - Vs_margin, np.amax(neuron.Vs) + Vs_margin]
Ws_margin = 0.1*(np.amax(neuron.Ws) - np.amin(neuron.Ws))
Ws_range = [np.amin(neuron.Ws) - Ws_margin, np.amax(neuron.Ws) + Ws_margin]
Vs = np.linspace(Vs_range[0], Vs_range[1], 300)

# Calculate the nullclines
Ws_Wnull = neuron.W_nullcline(Vs)
Ws_Vnull = neuron.V_nullcline(Vs, neuron.I_ext(ts[0]))


# Auxiliar function for the plot
def darken(color, val):
    """ Reduce the brightnesss of color to colorbrightness*val.
        
        The resulting color is given in hex format.
    
    """
    oldRGB = colors.to_rgb(color)
    oldhsv = colors.rgb_to_hsv(oldRGB)
    newhsv = np.array([oldhsv[0], oldhsv[1], val*oldhsv[2]])
    newrgb = colors.hsv_to_rgb(newhsv)
    newhex = colors.to_hex(newrgb)

    return newhex


# Create figures and axes
fig = plt.figure(figsize = (figxlen, figylen))
ax_phase = fig.add_subplot(121) 
ax_I = fig.add_subplot(224, xlim=(0, tmax))
ax_V = fig.add_subplot(222, xlim=(0, tmax), ylim=Vs_range, sharex=ax_I)
plt.setp(ax_V.get_xticklabels(), visible=False)
ax_phase.set_xlabel("V")
ax_phase.set_ylabel("W")
ax_V.set_ylabel("V")
ax_I.set_xlabel("Time")
ax_I.set_ylabel("I")

# Initialize plot
plot_Wnull, = ax_phase.plot(Vs, Ws_Wnull, linestyle="--", color="gray", 
                           label="W nullcline")
plot_Vnull, = ax_phase.plot(Vs, Ws_Vnull, linestyle="-.", color="gray",
                           label="V nullcline")

ax_I.plot(ts, neuron.I_ext(ts), color="black", visible=False)
plot_phase, = ax_phase.plot(neuron.Vs[:1], neuron.Ws[:1], color=traj_color)
plot_phasedot, = ax_phase.plot(neuron.Vs[:1], neuron.Ws[:1], linestyle="", 
                              marker="o", color=darken(traj_color, 0.75))
plot_V, = ax_V.plot(ts[0:1], neuron.Vs[0:1], color=plot_color)
plot_I, = ax_I.plot(ts[0:1], neuron.I_ext(ts[0:1]), color=plot_color)

ax_phase.legend()
fig.tight_layout()

# Create the animation
def update(i_anim, stepsperframe, ts, Vs, neuron, plot_phase, plot_phasedot,
           plot_V, plot_I, plot_Wnull, plot_Vnull):
    """Update function for the animation.

    """
    i = i_anim*stepsperframe

    # Recalculate
    Ws_Wnull = neuron.W_nullcline(Vs)
    Ws_Vnull = neuron.V_nullcline(Vs, neuron.I_ext(ts[i-1]))

    # Update plot
    plot_phase.set_data(neuron.Vs[:i], neuron.Ws[:i])
    plot_phasedot.set_data(neuron.Vs[i-1:i], neuron.Ws[i-1:i])
    plot_Wnull.set_data(Vs, Ws_Wnull)
    plot_Vnull.set_data(Vs, Ws_Vnull)
    plot_V.set_data(ts[:i], neuron.Vs[:i])
    plot_I.set_data(ts[:i], neuron.I_ext(ts[:i]))

    return plot_phase, plot_phasedot, plot_V, plot_I, plot_Wnull, plot_Vnull

points_per_second = int(animspeed/t_step)
points_per_frame = int(points_per_second/fps)
anim_interval = 1000./fps  # Interval between frames in ms
nframes = int(ts.size/points_per_frame)

anim = animation.FuncAnimation(fig, update, frames=nframes,
                               interval=anim_interval, blit=True,
                               fargs=(points_per_frame, ts, Vs, neuron,
                               plot_phase, plot_phasedot, plot_V, 
                               plot_I, plot_Wnull, plot_Vnull))
# Show plot
plt.show()

# Save the animation to file (this does not work when plt.show()
# has been used before)
# As mp4
#anim.save("phaseplaneI{0}_FN.mp4".format(I_ampl), dpi=200,
#        extra_args=['-vcodec', 'libx264'])
# As GIF (imagemagick must be installed)
#anim.save("phaseplaneI{0}_FN.gif".format(I_ampl), dpi=150,
#          writer='imagemagick')
