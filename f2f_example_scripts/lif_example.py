import brian2lava
from brian2 import *


def example(use_runtime = False):    
    set_device('lava', mode='preset', hardware='CPU', num_repr='fixed', use_f2f=True) # use preset-model mode with CPU (fixed-pt. representation)
    if use_runtime:
        set_device('runtime')
        prefs.codegen.target = 'numpy'
    from brian2lava.preset_mode.model_loader import lif
    # Set defaultclock
    defaultclock.dt = 0.5*ms

    # Define parameters
    N = 3
    bias = 4*mV/ms #0*mV/ms
    tau_j = 15*ms
    tau_v = 10*ms
    v_th = 50*mV
    v_rs = -5*mV
    ref = 0

    # Add a spike generator group
    S = SpikeGeneratorGroup(4, [0,1,2,0,2], [5,10,15,20,20]*ms)

    # Add a neuron group (which receives a background input)
    P = NeuronGroup(N, lif.equations, threshold=lif.conditions["th"], reset=lif.conditions["rs"], refractory='ref*ms', method='euler')
    P.j = "5*mV/ms"
    P.v = "45*mV"

    # Add another neuron group (which does not receive a background input)
    Q = NeuronGroup(N, lif.equations, threshold=lif.conditions["th"], reset=lif.conditions["rs"], refractory='ref*ms', method='euler')
    #Q.j = "0.1*mV/ms"
    Q.v = "2*mV"

    # Add synapses
    syn = Synapses(S, Q, model = 'w : volt/second', on_pre='j_post += w')
    syn.connect(i=[0,1,2], j=[1])
    syn.w = 5 * mV/ms

    # Add monitor for spikes
    #sm = SpikeMonitor(P)
    spmP = SpikeMonitor(P, variables='v', record=True)
    spmQ = SpikeMonitor(Q, variables='v', record=range(2))

    # Add monitors for voltage and current
    stmPv = StateMonitor(P, variables='v', record=True)
    stmPj = StateMonitor(P, variables='j', record=True)
    stmQv = StateMonitor(Q, variables='v', record=True)
    stmQj = StateMonitor(Q, variables='j', record=True)

    run(30*ms)

    return [spmP,spmQ,stmPv,stmPj,stmQv,stmQj]