import brian2lava
from brian2 import *


def example(use_runtime = False):
    # This must be run either way, because it initializes the model library!
    set_device('lava', mode='preset', hardware='CPU', num_repr='fixed', use_f2f=True) # use preset-model mode with CPU (fixed-pt. representation)
    if use_runtime:
        set_device('runtime')
        # Needed when switching from flexible mode to runtime 
        prefs.codegen.target = 'numpy'
        prefs.codegen.string_expression_target = 'numpy'
    from brian2lava.preset_mode.model_loader import probspiker

    # Define group of `probspiker` neurons
    N = 10
    freq = 50.0 * Hz
    defaultclock.dt = 0.5 * ms
    t_duration = 1000 * ms
    p_spike_0 = np.clip(freq * defaultclock.dt, 0, 1)  # spiking probability per time bin
    P = NeuronGroup(N, probspiker.equations,
                    threshold = probspiker.conditions['th'])
    P.p_spike = p_spike_0

    # Add monitors for spikes
    spmP = SpikeMonitor(P, variables='rnd', record=True)

    # Add monitor for rnd state
    stmP = StateMonitor(P, variables=['rnd', 'p_spike'], record=True)

    # Run the simulation
    run(t_duration)

    return [spmP,stmP]
