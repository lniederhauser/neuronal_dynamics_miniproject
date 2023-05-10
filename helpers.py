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
    
    # print(neuron)

    # initial values of v and w is set here:
    neuron.v = v_rest
    neuron.w = 0.0 * b2.pA

    # Monitoring membrane voltage (v) and w
    # state_monitor = b2.StateMonitor(neuron, ["v", "w", "I_stim"], record=True)
    # TODO: Find out if we have to record I_stim and how ??
    #  all state monitor can record are {'v', 'lastspike', 'not_refractory', 'w'}
    state_monitor = b2.StateMonitor(neuron, True, record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # running simulation
    b2.run(simulation_time)
    return state_monitor, spike_monitor


########################## DENDRITIC COMPARTMENT ################################
def simulate_dendritic(tau_d, C, v_rest, a, tau_w, E_d, D_d, g, I_stim, simulation_time=200 * b2.ms):
    r"""
    Simulation of the dendritic compartment using the following model equations

    The Brian2 model equations are:

    .. math::

        \frac{dv}{dt} = -\frac{v-v_rest}{tau_d} + \frac{I_stim+w+gf(v)}{C}
        \\
        \frac{dw}{dt} = \frac{-w+a(v-v_rest)}{tau_w}
        \\
        f(v) = \frac{1}{1+e^-\frac{v-E_d}{D})}

    Args:
        tau_d (Quantity): membrane time scale
        C (Quantity): membrane capacitance
        v_rest (Quantity): resting potential
        tau_w (Quantity): Adaptation time scale
        a (Quantity): factor modulating the adaptation variable equation
        E_d (Quantity): Sigmoid curve parameter
        D_d (Quantity): Sigmoid curve parameter
        g (Quantity): factor modulation the importance of the sigmoid in the voltage equation
        I_stim (TimedArray): Input current
        simulation_time (Quantity): Duration for which the model is simulated

    Returns:
        state_monitor:
        A b2.StateMonitor for the variables "v" and "w"
    """

    # EXP-IF
    eqs = """
        dv/dt = (-(v-v_rest)/tau_d) + (I_stim(t,i)+w+g*(1 /(1+exp(-(v-E_d)/D_d))))/C : volt
        dw/dt=(-w+a*(v-v_rest))/tau_w : amp
        """
    neuron = b2.NeuronGroup(1, model=eqs, method="euler")

    # initial values of v and w is set here:
    neuron.v = v_rest
    neuron.w = 0.0 * b2.pA

    # Monitoring membrane voltage (v) and w
    state_monitor = b2.StateMonitor(neuron, ["v", "w"], record=True)

    # running simulation
    b2.run(simulation_time)

    return state_monitor


def extract_rising_voltage_trace(state_monitor):
    voltage_array = b2.asarray(state_monitor.v)
    voltage_array = np.squeeze(voltage_array)
    id_max = voltage_array.argmax(axis=0)
    rising_voltage = voltage_array[:id_max]

    return rising_voltage


def compute_finite_derivative(voltage_trace):
    finite_derivative = np.zeros(voltage_trace.shape)
    finite_derivative[0] = voltage_trace[0]
    for i in range(1, voltage_trace.size):
        finite_derivative[i] = voltage_trace[i] - voltage_trace[i-1]

    return finite_derivative

######################################### TWO COMPARTMENT ######################################################
def simulate_pyramidal_neuron(tau_s, tau_d, C_s, C_d, v_rest, b, v_spike, tau_w_s, tau_w_d, I,
                              a, E_d, D_d, g_d, g_s, c_d, T_refractory, simulation_time=200 * b2.ms):
    r"""
    Simulation of the two-model compartment using model equations and refractory period

    Returns:
        (state_monitor, spike_monitor):
        A b2.StateMonitor for the variables "v" and "w" and a b2.SpikeMonitor
    """

    # EXP-IF
    # "unless refractory" arg allows voltage to stay unchanged during the refractory period

    zero_hz = 0 * b2.hertz
    zero_ms = 0 * b2.ms
    kernel_delay = 0.5 * b2.ms
    kernel_up_time = 2 * b2.ms

    eqs = """
        dv_s/dt = -(v_s - v_rest)/tau_s + (I(t,i) + w_s + g_s * f)/C_s : volt (unless refractory)
        dw_s/dt = -w_s/tau_w_s : amp
        dv_d/dt = -(v_d-v_rest)/tau_d + (I(t,i) + w_d + g_d * f + c_d * K)/C_d : volt
        dw_d/dt = (-w_d + a * (v_d - v_rest))/tau_w_d : amp
        dt_p_1/dt = 0 : second
        dt_p_2/dt = 0 : second
        dK/dt = zero_hz : 1
        f = 1 /(1 + exp(-(v_d-E_d)/D_d)) : 1
        """

    neuron = b2.NeuronGroup(1, model=eqs, threshold="v_s>v_spike", reset="v_s=v_rest;w_s+=b;v_d=v_d;w_d=w_d;t_p_1=t+kernel_delay", refractory=T_refractory,
                            method="euler", events={'K_step_up': '(t > t_p_1) and (t_p_1 != zero_ms)', 
                                                    'K_step_down': '(t > t_p_2) and (t_p_2 != zero_ms)'})
    neuron.run_on_event('K_step_up', 'K = 1; t_p_2 = t_p_1 + kernel_up_time; t_p_1 = zero_ms')
    neuron.run_on_event('K_step_down', 'K = 0; t_p_2 = zero_ms')
    
    # initial values of v and w is set here:
    neuron.v_s = v_rest
    neuron.w_s = 0.0 * b2.pA
    neuron.v_d = v_rest
    neuron.w_d = 0.0 * b2.pA
    neuron.t_p_1 = 0 * b2.ms
    neuron.t_p_2 = 0 * b2.ms
    neuron.K = 0

    state_monitor = b2.StateMonitor(neuron, ["v_s", "v_d", "w_s", "w_d", "K"], record=True)
    #spike_monitor = b2.SpikeMonitor(neuron)

    b2.run(simulation_time)
    return state_monitor#, spike_monitor


######################################### PLOTTING ######################################################
def plot_I_v_w(voltage_monitor, current, title=None, firing_threshold=None, legend_location=0, savefig = False):
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

    fig, ax = plt.subplots(3, 1, figsize=(10, 7))

    # Plot the input current I
    c = current(voltage_monitor.t, 0)
    ax[0].plot(voltage_monitor.t / b2.ms, c /b2.nA, "r", lw=2)
    ax[0].set_ylabel("Input current [nA]", fontsize=12)
    ax[0].grid()

    # Plot the voltage v
    ax[1].plot(time_values_ms, voltage_monitor[0].v / b2.mV, lw=2)
    if firing_threshold is not None:
        ax[1].plot(
            (voltage_monitor.t / b2.ms)[[0, -1]],
            [firing_threshold / b2.mV, firing_threshold / b2.mV],
            "r--", lw=2)
        ax[1].legend(["vm", "firing threshold"], fontsize=12, loc=legend_location)
        
    ax[1].set_ylabel("Membrane Voltage [mV]", fontsize=12)
    ax[1].grid()

    # Plot the adaptive term w
    ax[2].plot(time_values_ms, voltage_monitor[0].w /b2.nA, lw=2)
    ax[2].set_xlabel("t [ms]", fontsize=12)
    ax[2].set_ylabel("Adaption Variable [nA]", fontsize=12)
    ax[2].grid()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig("plots/" + title + ".png")
    plt.show()


def plot_voltage_derivative_curve(voltage, derivative, title=None, save_figure=False):
    
    plt.plot(voltage * 1000, derivative * 1000)
    plt.xlabel("Voltage [mV]", fontsize=12)
    plt.ylabel("Finite difference derivative [mV/ms]", fontsize=12)
    plt.grid()
    plt.title(title, fontsize=14)
    if save_figure:
        plt.savefig("plots/" + title + ".png")
    plt.show()


def plot_sigmoid(x, E, D, save_figure=False):
    E = float(E)
    D = float(D)
    y = 1/(1+np.exp(-(x-E)/D))
    plt.plot(x*1000, y)
    plt.grid()
    plt.xlabel("voltage [mV]")
    if save_figure:
        plt.savefig("plots/sigmoid.png")
    plt.show()



def plot_pyramidal(voltage_monitor, current, title=None, firing_threshold=None, legend_location=0, savefig = False):
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

    fig, ax = plt.subplots(6, 1, figsize=(10, 18))

    # Plot the input current I
    c = current(voltage_monitor.t, 0)
    ax[0].plot(voltage_monitor.t / b2.ms, c /b2.nA, "r", lw=2)
    ax[0].set_ylabel("Input current [nA]", fontsize=12)
    ax[0].grid()

    # Plot the Soma voltage v_s
    ax[1].plot(time_values_ms, voltage_monitor[0].v_s / b2.mV, lw=2)
    if firing_threshold is not None:
        ax[1].plot(
            (voltage_monitor.t / b2.ms)[[0, -1]],
            [firing_threshold / b2.mV, firing_threshold / b2.mV],
            "r--", lw=2)
        ax[1].legend(["vm", "firing threshold"], fontsize=12, loc=legend_location)
        
    ax[1].set_ylabel("Soma Membrane Voltage [mV]", fontsize=12)
    ax[1].grid()

    # Plot the Soma adaptive term w_s
    ax[2].plot(time_values_ms, voltage_monitor[0].w_s /b2.nA, lw=2)
    ax[2].set_ylabel("Soma Adaption Variable [nA]", fontsize=12)
    ax[2].grid()

    # Plot the dendrite voltage v_d
    ax[3].plot(time_values_ms, voltage_monitor[0].v_d /b2.mV, lw=2)
    ax[3].set_ylabel("Dendrite Membrane Voltage [mV]", fontsize=12)
    ax[3].grid()

    # Plot the dendrite adaptive term w_d
    ax[4].plot(time_values_ms, voltage_monitor[0].w_d /b2.nA, lw=2)
    ax[4].set_ylabel("Dendrite Adaption Variable [nA]", fontsize=12)
    ax[4].grid()

    # Plot the kernel K
    ax[5].plot(time_values_ms, voltage_monitor[0].K, lw=2)
    ax[5].set_xlabel("t [ms]", fontsize=12)
    ax[5].set_ylabel("Kernel (unitless)", fontsize=12)
    ax[5].grid()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig("plots/" + title + ".png")
    plt.show()