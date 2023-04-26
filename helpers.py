import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

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
    
    print(neuron)

    # initial values of v and w is set here:
    neuron.v = v_rest
    neuron.w = 0.0 * b2.pA

    # Monitoring membrane voltage (v) and w
    state_monitor = b2.StateMonitor(neuron, ["v", "w"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # running simulation
    b2.run(simulation_time)
    return state_monitor, spike_monitor



def plot_part_01(state_monitor, I, title=None):
    """Plots the state_monitor variables ["v", "I_e, "w"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    """

    #Plot Membrane Potential
    plt.subplot(311)
    plt.plot(state_monitor.t, state_monitor.v[0], lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.grid()

    #Plot the adaptation term w
    plt.subplot(312)
    plt.plot(state_monitor.t, state_monitor.w[0], "pink", lw=2)
    plt.xlabel("t (ms)")
    plt.ylabel("act./inact.")
    plt.legend(("w"))
    plt.ylim((0, 1))
    plt.grid()

    plt.xlabel("t [ms]")
    plt.ylabel("I [micro A]")
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    plt.show()


def plot_I_v_w(voltage_monitor, current, title=None, firing_threshold=None, legend_location=0):
    """plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current (TimedArray): injected current
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        legend_location (int): legend location. default = 0 (="best")

    Returns:
        the figure
    """

    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
    assert isinstance(current, b2.TimedArray), "current is not of type TimedArray"

    time_values_ms = voltage_monitor.t / b2.ms

    fig, ax = plt.subplots(3, 1)

    # Plot the input current I
    c = current(voltage_monitor.t, 0)
    ax[0].plot(voltage_monitor.t / b2.ms, c, "r", lw=2)
    ax[0].set_ylabel("Input current [A]")
    ax[0].grid()


    # Plot the voltage v
    ax[1].plot(time_values_ms, voltage_monitor[0].v / b2.mV, lw=2)
    if firing_threshold is not None:
        ax[1].plot(
            (voltage_monitor.t / b2.ms)[[0, -1]],
            [firing_threshold / b2.mV, firing_threshold / b2.mV],
            "r--", lw=2)
        ax[1].legend(["vm", "firing threshold"], fontsize=12, loc=legend_location)
        
    ax[1].set_ylabel("Membrane Voltage [mV]")
    ax[1].grid()


    #Plot the adaptive term w
    ax[2].plot(time_values_ms, voltage_monitor[0].w / b2.mV, lw=2)
    ax[2].set_xlabel("t [ms]")
    ax[2].set_ylabel("Adaption Variable [???]")
    ax[2].grid()


    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    #fig.savefig()