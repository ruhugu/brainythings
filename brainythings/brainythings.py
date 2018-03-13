#-*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import animation

class Neuron(class):  

    def __init__(self, g_leak=0.3, g_K=36, g_Na=120, V_leak=-54.402,
                 V_K=-77, V_Na=50):

        # Conductances (mS/cm2)
        self.g_leak = g_leak  # Leakage 
        self.g_K = g_K  # Potassium
        self.g_Na = g_Na  # Sodium

        # Membrane potentials for each ion (mV)
        self.V_leak = V_leak
        self.V_K = V_K
        self.V_Na = V_Na


class HHNeuron(Neuron):

    def __init__(self, neurondict=dict()):

        Neuron.__init__(self, **neurondict)


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
        beta = 4.*np.exp(-0.0556*(V + 60.)) 
        return beta

    def beta_h(self, V):
        beta = 1./(1 + np.exp(-0.1*(V + 35.)))
        return beta


    # Channel activation function time constant
    def _chanactiv_timeconst(V, alpha, beta):
        return 1./(alpha(V) + beta(V)

                
    #FIX
    def _chanactiv_diffrhs(V, alpha, beta):
        return 1./(alpha(V) + beta(V)
    
            chanactiv_timeconst = {
                "m": lambda V: self.chanactiv_timeconst(V, self.alpha_m, self.beta_m),
                "h": lambda V: self.chanactiv_timeconst(V, self.alpha_h, self.beta_h),
                "n": lambda V: self.chanactiv_timeconst(V, self.alpha_n, self.beta_n)}
    
            chanactiv_diffrhs = {
                "m": lambda V: self.chanactiv_diffrhs(V, self.alpha_m, self.beta_m),
                "h": lambda V: self.chanactiv_diffrhs(V, self.alpha_h, self.beta_h),
                "n": lambda V: self.chanactiv_diffrhs(V, self.alpha_n, self.beta_n)}
    
        
