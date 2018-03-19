#-*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)
import numpy as np
from scipy import integrate as spint
from matplotlib import pyplot as plt
from matplotlib import colors as colors
#from matplotlib import animation


class Neuron(object):  
    """Base neuron class.

    """
    def __init__(self, I_ampl=10., g_leak=0.3, 
                 g_K=36., g_Na=120., V_leak=-54.402, V_K=-77., V_Na=50.):

        # External current parameters (microA/cm2)
        self.I_ampl = I_ampl

        # Conductances (mS/cm2)
        self.g_leak = g_leak  # Leakage 
        self.g_K = g_K  # Potassium
        self.g_Na = g_Na  # Sodium

        # Membrane potentials for each ion (mV)
        self.V_leak = V_leak
        self.V_K = V_K
        self.V_Na = V_Na

        # Membrane capacity (microF/cm2)
        self.C = 1

        # Units
        self.I_unit = "(microA/cm2)"
        self.time_unit = "(ms)"
        self.V_unit = "(mV)"


    def I_ext(self, t):
        """External current function.

        """
        # Use np.ones() to accept arrays as input
        return self.I_ampl*np.ones(np.array(t).shape)


    def singleplot(self, y, label=None, figsize=3):
        """Plot varible y against time.

        """
        fig, ax = plt.subplots(figsize=(1.62*figsize, figsize))

        ax.plot(self.ts, y)
        ax.set_xlabel("Time {0}".format(self.time_unit))

        if label != None:
            ax.set_ylabel(label)

        fig.tight_layout()

        return fig



class HHNeuron(Neuron):
    def __init__(self, I_ampl=10., V_0=-65.,
                 m_0=None, n_0=None, h_0=None, neurondict=dict()):
        Neuron.__init__(self, I_ampl=I_ampl, **neurondict)

        # Note: Currents are given in microA/cm2, times in ms
        self.V_0 = V_0

        # Dictionaries with the corresponding functions for m, n and h
        self.ch_timeconst = {
                "m": (lambda V: self._ch_timeconst(V, self.alpha_m,
                                                          self.beta_m)),
                "h": (lambda V: self._ch_timeconst(V, self.alpha_h,
                                                          self.beta_h)),
                "n": (lambda V: self._ch_timeconst(V, self.alpha_n,
                                                          self.beta_n))}
        self.ch_asymp = {
                "m": (lambda V: self._ch_asymp(V, self.alpha_m,
                                                      self.beta_m)),
                "h": (lambda V: self._ch_asymp(V, self.alpha_h,
                                                      self.beta_h)),
                "n": (lambda V: self._ch_asymp(V, self.alpha_n,
                                                      self.beta_n))}
        self.chactiv_ddt = {
                "m": (lambda V, m: self._chactiv_ddt(V, m,
                                                    self.alpha_m, self.beta_m)),
                "h": (lambda V, h: self._chactiv_ddt(V, h, self.alpha_h,
                                                    self.beta_h)),
                "n": (lambda V, n: self._chactiv_ddt(V, n, self.alpha_n,
                                                    self.beta_n))}  

        # Initialize the channel activation functions to their 
        # saturation value
        if m_0 == None:
            self.m_0 = 0.05 #self.ch_asymp["m"](self.V_0)
        else:
            self.m_0 = m_0
        if n_0 == None:
            self.n_0 = 0.32 #self.ch_asymp["n"](self.V_0)
        else:
            self.n_0 = n_0
        if h_0 == None:
            self.h_0 = 0.6 #self.ch_asymp["h"](self.V_0)
        else:
            self.h_0 = h_0


    # Experimental data for potassium channels
    def alpha_n(self, V):
        alpha = 0.01*(V + 55.)/(1. - np.exp(-0.1*(V + 55.)))
        return alpha

    def beta_n(self, V):
        beta = 0.125*np.exp(-0.0125*(V + 65.)) 
        return beta

    # Experimental data for sodium channels
    def alpha_m(self, V):
        alpha = 0.1*(V + 40.)/(1. - np.exp(-0.1*(V + 40.)))
        return alpha
        
    def alpha_h(self, V):
        alpha = 0.07*np.exp(-0.05*(V + 65.))
        return alpha

    def beta_m(self, V):
        beta = 4.*np.exp(-0.0556*(V + 65.)) 
        return beta

    def beta_h(self, V):
        beta = 1./(1 + np.exp(-0.1*(V + 35.)))
        return beta

    # Functions
    def _ch_timeconst(self, V, alpha, beta):
        """Channel activation function time constant.

        """
        return 1./(alpha(V) + beta(V))

    def _ch_asymp(self, V, alpha, beta):
        """Asymptotic value of channel activation function.

        """
        return alpha(V)/(alpha(V) + beta(V))

    def _chactiv_ddt(self, V, chactiv, alpha, beta):
        """Time derivative of the chan. activation function.

        """
        timederivative = (self._ch_asymp(V, alpha, beta) 
                - chactiv)/self._ch_timeconst(V, alpha, beta)
#        timederivative = alpha(V)*(1. - chactiv) - beta(V)*chactiv
        return timederivative

    def ioncurrent(self, V, m, h, n):
        """Current due to the conduction of ions through the membrane channels.
        
        """
#        current = (self.g_leak*(V - self.V_leak) 
#                    + self.g_K*(n**4)*(V - self.V_K)
#                    + self.g_Na*h*(m**3)*(V - self.V_Na))
        current = self.I_leak(V) + self.I_K(V, n) + self.I_Na(V, h, m)
        return current

    def I_leak(self, V):
        """Leakeage current.

        """
        current = self.g_leak*(V - self.V_leak) 
        return current

    def I_K(self, V, n):
        """Ion current through potassium channels.

        """
        current = self.g_K*np.power(n, 4)*(V - self.V_K)
        return current
        
    def I_Na(self, V, h, m):
        """Ion current through sodium channels.

        """
        current = self.g_Na*h*np.power(m, 3)*(V - self.V_Na)
        return current

    def V_ddt(self, V, I_ext, m, h, n):
        """Time derivative of the membrane potential.
        
        """
        timederivative = (-self.ioncurrent(V, m, h, n) + I_ext)/self.C
        return timederivative
        
    def _rhs(self, y, t):
        """Right hand side of the system of equations to be solved.

        This functions is necessary to use scipy integrate.

        Parameters
        ----------
            y : array
                Array with the present state of the variables 
                which time derivative is to be solved: 
                (V, m, h, n)

            t : float
                Time variable.  

        Returns
        -------
            timederivatives : array
                Array with the time derivatives of the variables
                in the same order as y.
            
        """
        V = y[0]
        m = y[1]
        h = y[2]
        n = y[3]
        output = np.array((self.V_ddt(V, self.I_ext(t), m, h, n),
                           self.chactiv_ddt["m"](V, m),
                           self.chactiv_ddt["h"](V, h),
                           self.chactiv_ddt["n"](V, n)))
        return output


    def solve(self, ts=None):
        """Integrate the differential equations of the system.

        """
        # Simulation times
        if ts is None:
            self.ts = np.linspace(0, 200, 300)
        else:
            self.ts = ts

        y0 = np.array((self.V_0, self.m_0, self.h_0, self.n_0))
        sol = spint.odeint(self._rhs, y0, self.ts)
        self.Vs = sol[:,0]
        self.ms = sol[:,1]
        self.hs = sol[:,2]
        self.ns = sol[:,3]

        return


class FNNeuron(Neuron):
    """FitzHugh-Naguno neuron.

    The units in this model are different from the HH ones.

    Sources:
    https://en.wikipedia.org/w/index.php?title=FitzHugh%E2%80%93Nagumo_model&oldid=828788626
    http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

    """
    def __init__(self, I_ampl=0.85, V_0=-0.7, W_0 =-0.5, a=0.7, b=0.8,
                 tau=12.5, neurondict=dict()):
        Neuron.__init__(self, I_ampl=I_ampl, **neurondict)

        # Store intial conditions
        self.V_0 = V_0
        self.W_0 = W_0

        # Store model parameters
        self.a = a
        self.b = b
        self.tau = tau

        # Units
        self.time_unit = ""
        self.V_unit = ""
        self.I_unit = ""

        
    def V_ddt(self, V, W, I_ext):
        """Time derivative of the potential V.

        """
        timederivative = V - np.power(V, 3)/3. - W + I_ext
        return timederivative

    def W_ddt(self, V, W):
        """Time derivative of the recovery variable W.

        """
        timederivative = (V + self.a - self.b*W)/self.tau
        return timederivative

    def W_nullcline(self, V):
        """W value as a function of V in the W nullcline.

        Note: the W nullcline is the curve where the time derivative
        of W is zero.
        
        """
        return (V + self.a)/self.b

    def V_nullcline(self, V, I):
        """W value as a function of V in the V nullcline.
        
        Note: the V nullcline is the curve where the time derivative
        of V is zero.
        """
        return V - np.power(V, 3)/3. + I

        
    def _rhs(self, y, t):
        """Right hand side of the system of equations to be solved.

        This functions is necessary to use scipy integrate.

        Parameters
        ----------
            y : array
                Array with the present state of the variables 
                which time derivative is to be solved: 
                (V, W)

            t : float
                Time variable.  

        Returns
        -------
            timederivatives : array
                Array with the time derivatives of the variables
                in the same order as y.
            
        """
        V = y[0]
        W = y[1]
        output = np.array((self.V_ddt(V, W, self.I_ext(t)),
                           self.W_ddt(V, W)))
        return output


    def solve(self, ts=None):
        """Integrate the differential equations of the system.

        """
        # Simulation times
        if ts is None:
            self.ts = np.linspace(0, 1000, 1000)
        else:
            self.ts = ts

        y0 = np.array((self.V_0, self.W_0))
        sol = spint.odeint(self._rhs, y0, self.ts)
        # solve_ivp returns a lot of extra information about the solutions, but 
        # we are only interested in the values of the variables, which are stored
        # in sol.y
        self.Vs = sol[:,0]
        self.Ws = sol[:,1]

        return


# TODO: RENDIRSE Y APLICAR EULER
class IFlinNeuron(Neuron):
    """Linear integrate-and-fire neuron.

    Sources:
        http://icwww.epfl.ch/~gerstner/SPNM/node26.html
        http://www.ugr.es/~jtorres/Tema_4_redes_de_neuronas.pdf (spanish)

    """
    def __init__(self, I_ampl=10, V_0=-80, R=0.8, Vthr=-68.5,
                 neurondict=dict()):
        Neuron.__init__(self, I_ampl=I_ampl, **neurondict)

        # Store initial condition
        self.V_0 = V_0

        # Store parameters
        self.R = R  # kohmn/cm2
        self.Vthr = Vthr  # Fire threshold


    def V_ddt(self, V, I_ext):
        """Time derivative of the membrane potential.

        """
        timederivative = (-(V + 65)/self.R + I_ext)/self.C
        return timederivative


    def _thresholddiff(self, t, V):
        """Difference between the membrane potential and the fire threshold.

        This is used as an auxiliary function in the solver to 
        find when the pulse starts.

        """
        diff = V - self.Vthr
        return diff


    def _rhs(self, t, y):
        """Right hand side of the system of equations to be solved.

        This functions is necessary to use scipy.integrate.

        Parameters
        ----------
            y : float
                Array with the present state of the variable
                which time derivative is to be solved, V.

            t : float
                Time variable.  

        Returns
        -------
            timederivative : float
                Time derivatives of the variable.
            
        """
        V = y
        output = self.V_ddt(V, self.I_ext(t))
        return output

    def solve(self, ts=None):
        """Integrate the differential equations of the system.

        """
        if ts is None:
            self.ts = np.linspace(0, 100, 400)
        else:
            self.ts = ts

        # Empty array to store the results
        self.Vs = np.zeros(ts.size)

        # Initial value
        y0 = np.array((self.V_0,))

        def thresholddiff(t, V):
            """Difference between the potential and the fire threshold.

            This is used as an auxiliary function in the solver to 
            find when the pulse starts

            """
            diff = self.Vthr - V
            return diff
        thresholddiff.terminal = True

        sol = spint.solve_ivp(self._rhs, (ts[0], ts[-1]), y0,
                              events=thresholddiff, dense_output=True)

        # Index in ts of next point to be calculated
        next_idx = 0

        # Store the calculated points
        npoints = sol.t.size
        self.Vs[next_idx, next_idx + npoints] = sol.y[0]
        lastidx += npoints

        # Repeat until 
        while sol,status == 1:
            self.Vs = sol.y[0]
        return sol

#    def solve(self, ts=None):
#        """Integrate the differential equations of the system.
#
#        """
#        # Simulation times
#        if ts is None:
#            self.ts = np.linspace(0, 1000, 1000)
#        else:
#            self.ts = ts
#
#        y0 = self.V_0
#        sol = spint.odeint(self._rhs, y0, self.ts)
#        # solve_ivp returns a lot of extra information about the solutions, but 
#        # we are only interested in the values of the variables, which are stored
#        # in sol.y
#        self.Vs = sol[:,0]
#
#        return

