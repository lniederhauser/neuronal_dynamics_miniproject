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
    """Extract the part of the voltage trace from the beginning until it reaches its maximum

        Args:
            state_monitor (StateMonitor): recorded voltage

        Returns:
            an array containing the recorded voltage up until it reaches its max
    """
    voltage_array = b2.asarray(state_monitor.v)
    voltage_array = np.squeeze(voltage_array)
    id_max = voltage_array.argmax(axis=0)
    rising_voltage = voltage_array[:id_max]

    return rising_voltage


def compute_finite_derivative(voltage_trace):
    """Computes the finite derivative of the voltage trace, i.e. dV(t) = V(t)-V(t-1) for each time t. With dV(0) = V(0)

        Args:
            voltage_trace (StateMonitor): recorded voltage

        Returns:
            an array containing the finite derivative of voltage_trace
        """
    finite_derivative = np.zeros(voltage_trace.shape)
    finite_derivative[0] = voltage_trace[0]
    for i in range(1, voltage_trace.size):
        finite_derivative[i] = voltage_trace[i] - voltage_trace[i-1]

    return finite_derivative

######################################### TWO COMPARTMENTS ######################################################


def simulate_pyramidal_neuron(tau_s, tau_d, C_s, C_d, v_rest, b, v_spike, tau_w_s, tau_w_d, I_s, I_d,
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
            dv_s/dt = -(v_s - v_rest)/tau_s + (I_s(t,i) + w_s + g_s * f)/C_s : volt (unless refractory)
            dw_s/dt = -w_s/tau_w_s : amp
            dv_d/dt = -(v_d-v_rest)/tau_d + (I_d(t,i) + w_d + g_d * f + c_d * K)/C_d : volt
            dw_d/dt = (-w_d + a * (v_d - v_rest))/tau_w_d : amp
            dt_p/dt = 0 : second
            dK/dt = zero_hz : 1
            f = 1 /(1 + exp(-(v_d-E_d)/D_d)) : 1
            """
    neuron = b2.NeuronGroup(1, model=eqs, threshold="v_s>v_spike", reset="v_s=v_rest;w_s+=b;t_p=t",
                            refractory=T_refractory, method="euler",
                            events={'K_step_up': '(t >= t_p+kernel_delay) and (t < t_p+kernel_delay+kernel_up_time)',
                                    'K_step_down': 't >= t_p+kernel_delay+kernel_up_time'})
    neuron.run_on_event('K_step_up', 'K = 1')
    neuron.run_on_event('K_step_down', 'K = 0')
    
    # initial values of v and w is set here:
    neuron.v_s = v_rest
    neuron.w_s = 0.0 * b2.pA
    neuron.v_d = v_rest
    neuron.w_d = 0.0 * b2.pA
    # neuron.t_p_1 = 0 * b2.ms
    # neuron.t_p_2 = 0 * b2.ms
    neuron.t_p = float("inf") * b2.ms  # initiate to a high value so that t !> (t_p + kernel delay)
    neuron.K = 0

    state_monitor = b2.StateMonitor(neuron, ["v_s", "v_d", "w_s", "w_d", "K"], record=True)
    #spike_monitor = b2.SpikeMonitor(neuron)

    b2.run(simulation_time)
    return state_monitor#, spike_monitor


######################################### TWO COMPARTMENTS WITH NOISE ##################################################
def simulate_pyramidal_neuron_noisy(tau_s, tau_d, tau_ou, mu_s, mu_d, sigma_ou, C_s, C_d, v_rest, b, v_spike, tau_w_s,
                                    tau_w_d, I_s_ext, I_d_ext, a, E_d, D_d, g_d, g_s, c_d, T_refractory,
                                    simulation_time=200*b2.ms, nb_neurons=1):
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
            dv_s/dt = -(v_s - v_rest)/tau_s + (I_s + w_s + g_s * f)/C_s : volt (unless refractory)
            dw_s/dt = -w_s/tau_w_s : amp
            dv_d/dt = -(v_d-v_rest)/tau_d + (I_d + w_d + g_d * f + c_d * K)/C_d : volt
            dw_d/dt = (-w_d + a * (v_d - v_rest))/tau_w_d : amp
            dt_p/dt = 0 : second
            dK/dt = zero_hz : 1
            f = 1 /(1 + exp(-(v_d-E_d)/D_d)) : 1
            dI_s_bg/dt = (mu_s - I_s_bg)/tau_ou + sqrt(2/tau_ou)*sigma_ou*xi_1 : amp
            dI_d_bg/dt = (mu_d - I_d_bg)/tau_ou + sqrt(2/tau_ou)*sigma_ou*xi_2 : amp
            I_s = I_s_ext(t) + I_s_bg : amp
            I_d = I_d_ext(t) + I_d_bg : amp
            """
    neuron = b2.NeuronGroup(N=nb_neurons, model=eqs, threshold="v_s>v_spike", reset="v_s=v_rest;w_s+=b;t_p=t",
                            refractory=T_refractory,
                            events={'K_step_up': '(t >= t_p+kernel_delay) and (t < t_p+kernel_delay+kernel_up_time)',
                                    'K_step_down': 't >= t_p+kernel_delay+kernel_up_time'})
    neuron.run_on_event('K_step_up', 'K = 1')
    neuron.run_on_event('K_step_down', 'K = 0')
    
    # initial values of v and w is set here:
    neuron.v_s = v_rest
    neuron.w_s = 0.0 * b2.pA
    neuron.v_d = v_rest
    neuron.w_d = 0.0 * b2.pA
    # neuron.t_p_1 = 0 * b2.ms
    # neuron.t_p_2 = 0 * b2.ms
    neuron.t_p = float("inf") * b2.ms  # initiate to a high value so that t !> (t_p + kernel delay)
    neuron.K = 0

    state_monitor = b2.StateMonitor(neuron, ["v_s", "v_d", "w_s", "w_d", "K", "I_s_bg", "I_d_bg", "I_s", "I_d"], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    b2.run(simulation_time)
    return state_monitor, spike_monitor


'''
def compute_firing_rate(spike_monitor, sim_time=800, time_step=1 * b2.ms):
    spike_times = np.array(spike_monitor.t / b2.ms)  # When each spike occurs in order of time
    spike_neurons = np.array(spike_monitor.i)  # Which neuron fires at the spike time
    time = np.arange(sim_time)
    t_dig = np.digitize(t, time)
    values = np.unique(t_dig)
    firing_rate = np.zeros(sim_time)
    for v in values:
        firing_rate[v - 1] += (t_dig == v).sum()
    firing_rate = firing_rate / (sim_time * time_step)

    return firing_rate
'''


def compute_spike_and_burst_scatter(spike_monitor, nb_neurons, sim_time):

    spike_times = np.array(spike_monitor.t / b2.ms)  # When each spike occurs in order of time
    spike_neurons = np.array(spike_monitor.i)  # Which neuron fires at the spike time
    time_points = np.arange(sim_time)  # 800 time points acting here as 1ms bins
    spike_times_dig = np.digitize(spike_times, time_points)  # Digitized spike times

    # Creating null 2D array (neurons, spike time) with 1 only where a spike occurred
    spike_scatter = np.zeros((nb_neurons, sim_time))
    for s, n in zip(spike_times_dig, spike_neurons):
        spike_scatter[int(n - 1), int(s - 1)] = 1

    # Creating a null 2D array (neurons, spike time) with 1 only where a burst occurred
    burst_scatter = np.zeros(spike_scatter.shape)
    rect_16 = np.ones(16)
    for n in range(nb_neurons):
        burst_scatter[n, :] = np.where(np.convolve(spike_scatter[n, :], rect_16, 'same') > 1, 1, 0)

    return spike_scatter, burst_scatter


def compute_firing_and_burst_rate(spike_monitor, nb_neurons=4000, sim_time=800):

    spike_scatter, burst_scatter = compute_spike_and_burst_scatter(spike_monitor, nb_neurons, sim_time)

    # Mean and time normalisation to get rates
    firing_rate = np.mean(spike_scatter, axis=0) / b2.ms
    burst_rate = np.mean(burst_scatter, axis=0) / b2.ms

    return firing_rate, burst_rate


def compute_event_rate(spike_monitor, burst_rate, nb_neurons=4000, sim_time=800):

    spike_scatter, burst_scatter = compute_spike_and_burst_scatter(spike_monitor, nb_neurons, sim_time)

    # Creating a null 2D array (neurons, spike time) with 1 only where an isolated single spike occurs (i.e. the spike
    # is not part of a burst
    single_spike_scatter = np.logical_and(spike_scatter, np.logical_not(burst_scatter))

    # Mean and time normalisation to get rate
    single_spike_rate = np.mean(single_spike_scatter, axis=0) / b2.ms

    event_rate = single_spike_rate + burst_rate

    return event_rate


def compute_burst_proba(burst_rate, event_rate):
    proba = np.zeros(event_rate.shape)
    for i, rate in enumerate(event_rate):
        if rate !=0:
            proba[i] = burst_rate[i]/rate

    return proba


############################################## INPUT CURRENTS  #########################################################
def get_EPSC_current(t_start, t_end, unit_time, amplitude, tau, append_zero=True):
    """Creates an Excitatory Post-Synaptic Current (EPSC) shaped current

    Args:
        t_start (int): start of the EPSC current
        t_end (int): end of the EPSC current
        unit_time (Quantity, Time): unit of t_start and t_end. e.g. 0.1*brian2.ms
        amplitude (Quantity, Current): maximum amplitude of EPSC current
        tau (Quantity, Time): decay rate of the EPSC
        append_zero (bool, optional): if true, 0Amp is appended at t_end+1. Without that
            trailing 0, Brian reads out the last value in the array for all indices > t_end.


    Returns:
        TimedArray: Brian2.TimedArray
    """
    assert isinstance(t_start, int), "t_start_ms must be of type int"
    assert isinstance(t_end, int), "t_end must be of type int"
    assert b2.units.fundamentalunits.have_same_dimensions(amplitude, b2.amp), \
        "amplitude must have the dimension of current. e.g. brian2.uamp"
    assert b2.units.fundamentalunits.have_same_dimensions(tau, b2.ms), \
        "decay rate must have the dimension of time. e.g. brian2.ms"

    tmp_size = 1 + t_end  # +1 for t=0
    if append_zero:
        tmp_size += 1
    tmp = np.zeros((tmp_size, 1)) * b2.amp
    if t_end > t_start:  # if deltaT is zero, we return a zero current
        t = range(0, (t_end - t_start) + 1) * unit_time
        exp_decay = np.exp(-t/tau)
        c = amplitude*(1-exp_decay)*exp_decay
        tmp[t_start: t_end + 1, 0] = c
    curr = b2.TimedArray(tmp, dt=1. * unit_time)
    return curr


def get_alternating_current(t_start, t_end, unit_time, high_current, low_current, t_down, t_up, unit_current=b2.pA,
                            phase_lag=0, append_zero=False):
    assert isinstance(t_start, int), "t_start must be of type int"
    assert isinstance(t_end, int), "t_end must be of type int"
    assert isinstance(t_down, int), "t_down must be of type int"
    assert isinstance(t_up, int), "t_up must be of type int"
    assert b2.units.fundamentalunits.have_same_dimensions(high_current, b2.amp), \
        "high_current must have the dimension of current. e.g. brian2.uamp"
    assert b2.units.fundamentalunits.have_same_dimensions(low_current, b2.amp), \
        "low_current must have the dimension of current. e.g. brian2.uamp"

    tmp_size = 1 + t_end  # +1 for t=0
    if append_zero:
        tmp_size += 1
    tmp = np.zeros(tmp_size) * unit_current
    tmp[t_start:t_start + phase_lag] = low_current
    if t_end > t_start:  # if deltaT is zero, we return a zero current
        time_array = range(0, (t_end - t_start - phase_lag) + 1)
        period = t_down + t_up
        period_counter = 0
        c = np.zeros(len(time_array))
        for i, t in enumerate(time_array):
            if t-(period_counter*period) <= t_up:
                c[i] = high_current / unit_current
            elif t-period_counter*period > t_up:
                c[i] = low_current / unit_current
            if t >= period*(period_counter+1):
                period_counter += 1
        tmp[t_start+phase_lag: t_end + 1] = c * unit_current
    curr = b2.TimedArray(tmp, dt=1.*unit_time)
    return curr


######################################### PLOTTING ######################################################
def plot_I_v_w(voltage_monitor, current, title=None, firing_threshold=None, legend_location=0, savefig=False):
    """plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current (TimedArray): injected current
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        legend_location (int): legend location. default = 0 (="best")
        savefig (Bool): If True, the figure is saved in the directory /plots, default = False
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
    """plots the voltage wrt its derivative

        Args:
            voltage (array): voltage
            derivative (array): derivative of the voltage
            title (string, optional): title of the figure
            save_figure (Bool, optional): If True, the figure is saved in the directory /plots, default = False
    """
    plt.plot(voltage * 1000, derivative * 1000)
    plt.xlabel("Voltage [mV]", fontsize=12)
    plt.ylabel("Finite difference derivative [mV/ms]", fontsize=12)
    plt.grid()
    plt.title(title, fontsize=14)
    if save_figure:
        plt.savefig("plots/" + title + ".png")
    plt.show()


def plot_sigmoid(x, E, D, save_figure=False):
    """plots a sigmoid using the following equation
    .. math::
        f(x) = \frac{1}{1+e^-\frac{x-E}{D})}

        Args:
            x (array): values for which the sigmoid is plotted
            E, D (float): parameters that determine the shape of the sigmoid
            save_figure (Bool): If True, the figure is saved in the directory /plots, default = False

    """
    E = float(E)
    D = float(D)
    y = 1/(1+np.exp(-(x-E)/D))
    plt.plot(x*1000, y)
    plt.grid()
    plt.xlabel("voltage [mV]")
    if save_figure:
        plt.savefig("plots/sigmoid.png")
    plt.show()


def plot_EPSC_current(current, unit_amp, unit_time, title=None):
    """plots a current wrt time, with the x-axis in ms and the y-axis in nA
        Args:
            current (2d TimedArray): the current to plot
            unit_amp (Quantity, Voltage): unit of the voltage of the  current, e.g. brian2.nA
            unit_time (Quantity, Time): unit of time used to generate the current e.g. 0.1*brian2.ms
            title (string, optional): title of the figure

    """
    y = current.values[:, 0] / unit_amp
    x = range(0, len(current.values[:, 0]))*unit_time
    plt.plot(x*1000, y)
    plt.title(title)
    plt.xlabel("time [ms]")
    plt.ylabel("current [nA]")
    plt.grid()
    plt.show()


def plot_pyramidal(voltage_monitor, current_s, current_d, title=None, firing_threshold=None, legend_location=0, savefig=False):
    """plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current_s (TimedArray): current injected into the soma
        current_d (TimedArray): current injected into the dendrite
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        legend_location (int): legend location. default = 0 (="best")
        savefig (bool): If True, the figure is saved in the directory /plots, default = False

    """

    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
    assert isinstance(current_s, b2.TimedArray), "somatic current is not of type TimedArray"
    assert isinstance(current_d, b2.TimedArray), "dendritic current is not of type TimedArray"

    time_values_ms = voltage_monitor.t / b2.ms

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    # Plot the input currents of the soma
    c = current_s(voltage_monitor.t, 0)
    ax[0, 0].plot(voltage_monitor.t / b2.ms, c /b2.nA, lw=2, label="soma")
    # Plot input current of dendrite
    c_d = current_d(voltage_monitor.t, 0)
    ax[0, 0].plot(voltage_monitor.t/b2.ms, c_d/b2.nA, label="dendrite")
    ax[0, 0].set_ylabel("Input current [nA]", fontsize=12)
    ax[0, 0].set_xlabel("t [ms]", fontsize=12)
    ax[0, 0].legend()
    ax[0, 0].grid()

    # Plot the kernel K
    ax[0, 1].plot(time_values_ms, voltage_monitor[0].K, c="r", lw=2)
    ax[0, 1].set_xlabel("t [ms]", fontsize=12)
    ax[0, 1].set_ylabel("Kernel (unitless)", fontsize=12)
    ax[0, 1].grid()

    # Plot the Soma voltage v_s
    ax[1, 0].plot(time_values_ms, voltage_monitor[0].v_s / b2.mV, lw=2, label="soma")
    if firing_threshold is not None:
        ax[1, 0].plot(
            (voltage_monitor.t / b2.ms)[[0, -1]],
            [firing_threshold / b2.mV, firing_threshold / b2.mV],
            "r--", lw=2)
        ax[1, 0].legend(["vm", "firing threshold"], fontsize=12, loc=legend_location)
    # Plot the dendrite voltage v_d
    ax[1, 0].plot(time_values_ms, voltage_monitor[0].v_d / b2.mV, lw=2, label="dendrite")
    ax[1, 0].set_ylabel("Membrane Voltage [mV]", fontsize=12)
    ax[1, 0].set_xlabel("t [ms]", fontsize=12)
    ax[1, 0].legend()
    ax[1, 0].grid()

    # Plot the Soma adaptive term w_s
    ax[1, 1].plot(time_values_ms, voltage_monitor[0].w_s /b2.nA, lw=2, label="soma")
    # Plot the dendrite adaptive term w_d
    ax[1, 1].plot(time_values_ms, voltage_monitor[0].w_d /b2.nA, lw=2, label="dendrite")
    ax[1, 1].set_ylabel("Adaption Variable [nA]", fontsize=12)
    ax[1, 1].set_xlabel("t [ms]", fontsize=12)
    ax[1, 1].legend()
    ax[1, 1].grid()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig("plots/" + title + ".png")
    plt.show()


def plot_noise_and_noisy_currents(voltage_monitor, title=None, savefig=False):
    """plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current_s (TimedArray): current injected into the soma
        current_d (TimedArray): current injected into the dendrite
        title (string, optional): title of the figure
        legend_location (int): legend location. default = 0 (="best")
        savefig (bool): If True, the figure is saved in the directory /plots, default = False

    """

    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"

    time_values_ms = voltage_monitor.t / b2.ms

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    # Plot the current at the soma
    ax[0, 0].plot(voltage_monitor.t / b2.ms, voltage_monitor[0].I_s /b2.nA, lw=2)
    ax[0, 0].set_ylabel("Current [nA]", fontsize=12)
    ax[0, 0].set_xlabel("t [ms]", fontsize=12)
    ax[0, 0].set_title("Total Current at Soma")
    ax[0, 0].grid()

    # Plot the current at the dendrite
    ax[0, 1].plot(voltage_monitor.t / b2.ms, voltage_monitor[0].I_d /b2.nA, lw=2)
    ax[0, 1].set_ylabel("Current [nA]", fontsize=12)
    ax[0, 1].set_xlabel("t [ms]", fontsize=12)
    ax[0, 1].set_title("Total Current at Dendrite")
    ax[0, 1].grid()

    # Plot the current noise at the soma
    ax[1, 0].plot(voltage_monitor.t / b2.ms, voltage_monitor[0].I_s_bg /b2.nA, lw=2)
    ax[1, 0].set_ylabel("Noise [nA]", fontsize=12)
    ax[1, 0].set_xlabel("t [ms]", fontsize=12)
    ax[1, 0].set_title("Noise at Soma")
    ax[1, 0].grid()

    # Plot the current noise at the dendrite
    ax[1, 1].plot(voltage_monitor.t / b2.ms, voltage_monitor[0].I_s_bg /b2.nA, lw=2)
    ax[1, 1].set_ylabel("Noise [nA]", fontsize=12)
    ax[1, 1].set_xlabel("t [ms]", fontsize=12)
    ax[1, 1].set_title("Noise at Dendrite")
    ax[1, 1].grid()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig("plots/" + title + ".png")
    plt.show()


def plot_alternating_currents(current1, current2, label1, label2, unit_time=b2.ms, unit_amp=b2.pA, title=None):
    plt.plot(range(0, len(current1.values)) * unit_time * 1000, current1.values / unit_amp, label=label1)
    plt.plot(range(0, len(current2.values)) * unit_time * 1000, current2.values / unit_amp, label=label2)
    plt.title(title)
    plt.xlabel("time [ms]")
    plt.ylabel("current [pA]")
    plt.legend()
    plt.grid()
    plt.show()


def plot_external_inputs_and_rates(firing_rate, bursting_rate, soma_current, dendrite_current,
                                   isBurstProba=False, title=None, savefig=False):

    # Smoothing with convolution and 10ms window
    rect_10 = np.ones(10)
    firing_rate = np.convolve(firing_rate, rect_10, 'same') / 10
    bursting_rate = np.convolve(bursting_rate, rect_10, 'same') / 10

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    if isBurstProba:
        label0a = "event rate"
        label1a = "burst" + '\n' + "probability"
        ylabel0a = "Event rate [Hz]"
        title0 = "Population smooth event activity"
        ylabel1a = "Bursting probability"
        title1 = "Population smooth bursting probability"
    else:
        label0a = "firing rate"
        label1a = "bursting" + '\n' + "rate"
        ylabel0a = 'Firing rate [Hz]'
        title0 = "Population smooth Firing Activity"
        ylabel1a = 'Bursting rate [Hz]'
        title1 = 'Population Smooth Bursting Activity'

    ax0b = ax[0].twinx()
    ax[0].plot(firing_rate, label=label0a)
    ax[0].set_xlabel('Time [ms]')
    ax[0].set_ylabel(ylabel0a)
    ax0b.plot(range(0, len(soma_current.values)) * b2.ms * 1000, soma_current.values / b2.pA,
              label="soma" + '\n' + "input current", c='orange')
    ax0b.set_ylabel('Input current [pA]')
    ax[0].set_title(title0 + ' and external somatic stimulation')
    ax[0].grid()
    ax[0].legend(loc='best')
    ax0b.legend(loc='lower right')

    ax1b = ax[1].twinx()
    ax1b.plot(range(0, len(dendrite_current.values)) * b2.ms * 1000, dendrite_current.values / b2.pA,
              label="dendrite" + '\n' + "input current", c='orange')
    ax1b.set_ylabel('Input current [pA]')
    ax[1].plot(bursting_rate, label=label1a)
    ax[1].set_xlabel('Time [ms]')
    ax[1].set_ylabel(ylabel1a)
    ax[1].set_title(title1 + ' and external dendritic stimulation')
    ax[1].grid()
    ax[1].legend(loc='upper left')
    ax1b.legend(loc='lower left')

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig("plots/" + title + ".png")
    plt.show()
