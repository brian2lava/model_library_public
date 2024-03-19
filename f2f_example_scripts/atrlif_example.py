from brian2 import *
import brian2lava

def example(use_runtime = False):
    # This must be run either way, because it initializes the model library!
    set_device('lava', mode='preset', hardware='CPU', num_repr='fixed', use_f2f=True) # use preset-model mode with CPU (fixed-pt. representation)
    if use_runtime:
        set_device('runtime')
        # Needed when switching from flexible mode to runtime 
        prefs.codegen.target = 'numpy'
        prefs.codegen.string_expression_target = 'numpy'
    from brian2lava.preset_mode.model_loader import atrlif
    defaultclock.dt = 0.5*ms
    # Define parameters (same as for the example in our internal neuron models repo)
    N = 3
    bias = 3 * mV/ms
    tau_j = 1 * ms
    tau_v = 1 / 0.4 * ms #10 *   ms # delta_v = dt/tau_v*2^12
    tau_theta = 1 / 0.4 * ms #10 *   ms
    tau_r = 1 / 0.2 * ms #5 *   ms
    theta_0 = 5 *   mV
    theta_step = 3.75 * mV

    # Transcribe reset conditions (convert list to multiline string)
    reset_cond = "\n".join(atrlif.conditions["rs"])

    # Add a spike generator group
    S = SpikeGeneratorGroup(N, [0, 0], [2, 6]*ms) 

    # Add a neuron group (which receives a background input)
    P = NeuronGroup(N, atrlif.equations, threshold=atrlif.conditions["th"], reset=reset_cond, method='euler')
    P.j = "0 * mV/ms"
    P.v = "0 * mV"
    P.theta = theta_0
    P.r = "0 * mV"

    # Add another neuron group (which does not receive a background input)
    Q = NeuronGroup(N, atrlif.equations, threshold=atrlif.conditions["th"], reset=reset_cond, method='euler')
    #Q.j = "0.1*mV/ms"
    Q.v = "0 * mV"
    Q.theta = theta_0
    Q.r = "0 * mV"

    # Add synapses
    syn = Synapses(S, Q, model = 'w : volt/second', on_pre='j_post += w')
    syn.connect(i=[0], j=[0,1,2])
    syn.w = 60 * mV/ms

    # Add monitor for spikes
    #sm = SpikeMonitor(P)
    spmP = SpikeMonitor(P, variables='v', record=True) # TODO adapt to adaptive-thr.
    spmQ = SpikeMonitor(Q, variables='v', record=False) # TODO adapt to adaptive-thr.

    # Add monitors for voltage and current
    stmPv = StateMonitor(P, variables='v', record=True)
    stmPj = StateMonitor(P, variables='j', record=True)
    stmPtheta = StateMonitor(P, variables='theta', record=True)
    stmPr = StateMonitor(P, variables='r', record=True)
    stmQv = StateMonitor(Q, variables='v', record=True)
    stmQj = StateMonitor(Q, variables='j', record=True)
    stmQtheta = StateMonitor(Q, variables='theta', record=True)
    stmQr = StateMonitor(Q, variables='r', record=True)

    run(21 * ms)

    return [stmPv,stmPj,stmPtheta,stmPr,stmQv,stmQj,stmQtheta,stmQr]