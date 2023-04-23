import brian2 as b2


########################## SOMA COMPARTMENT ################################


def simulate_soma(tau_s, C, v_rest, b, v_spike, tau_w, I_stim, T_refractory, simulation_time=200 * b2.ms):
    r"""
    Simulation of the soma compartment using model equations and refractory period

    The Brian2 model equations are:

    .. math::

        \frac{dv}{dt} = -\frac{v-v_rest}{tau_s} + \frac{I_stim+w}{C}
        \\
        \frac{dw}{dt} = \frac{-w}{tau_w}

    Args:
        tau_s (Quantity): membrane time scale
        C (Quantity): membrane capacitance
        v_rest (Quantity): resting potential
        b (Quantity): Spike-triggered adaptation current (=increment of w after each spike)
        v_spike (Quantity): voltage threshold for the spike condition
        tau_w (Quantity): Adaptation time scale
        I_stim (TimedArray): Input current
        T_refractory (Quantity): Refractory period
        simulation_time (Quantity): Duration for which the model is simulated

    Returns:
        (state_monitor, spike_monitor):
        A b2.StateMonitor for the variables "v" and "w" and a b2.SpikeMonitor
    """

    # EXP-IF
    # "unless refractory" arg allows voltage to stay unchanged during the refractory period
    eqs = """
        dv/dt = (-(v-v_rest)/tau_s) + ((I_stim(t,i) + w)/C) : volt (unless refractory)
        dw/dt=-w/tau_w : amp
        """

    neuron = b2.NeuronGroup(1, model=eqs, threshold="v>v_spike", reset="v=v_rest;w+=b", refractory=T_refractory,
                            method="euler")

    # initial values of v and w is set here:
    neuron.v = v_rest
    neuron.w = 0.0 * b2.pA

    # Monitoring membrane voltage (v) and w
    state_monitor = b2.StateMonitor(neuron, ["v", "w"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # running simulation
    b2.run(simulation_time)
    return state_monitor, spike_monitor
