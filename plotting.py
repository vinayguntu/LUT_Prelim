import matplotlib.pyplot as plt
import numpy as np

def plot_figure(means, stdevs, n_steps, dt, tstep=100, fbmod=None, savefig=True):
    #Plot bladder volume and bladder pressure
    if fbmod is not None:
        fig1, ax1_1 = plt.subplots()

        color = 'tab:red'
        ax1_1.set_xlabel('Time (t) [ms]')
        # ax1_1.set_ylabel('Bladder Volume (V) [ml]', color=color)
        ax1_1.plot(fbmod.times, fbmod.b_vols, color=color)
        ax1_1.tick_params(axis='y', labelcolor=color)

        ax2_1 = ax1_1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2_1.set_ylabel('Bladder Pressure (P) [cm H20]', color=color)  # we already handled the x-label with ax1
        ax2_1.plot(fbmod.times, fbmod.b_pres, color=color)
        ax2_1.tick_params(axis='y', labelcolor=color)

        fig1.tight_layout()  # otherwise the right y-label is slightly clipped

    # tstep (ms)
    tstop = (n_steps-1)*dt
    t = np.arange(0.0,tstop,tstep)
    ind = np.floor(t/dt).astype(np.int)

    fig2 = plt.figure()
    plt.plot(t, means['Bladaff'][ind], color='b', marker='^', mfc='b', mec='b', label='Bladder Afferent')
    plt.plot(t, means['PGN'][ind], color='g', marker='o', mfc='g', mec='g', label='PGN')
    plt.plot(t, means['EUSmn'][ind], color='k', marker='D', mfc='k', mec='k', label='EUS Motor Neurons')
    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()

    fig3 = plt.figure()
    plt.plot(t, means['INmminus'][ind], color='b', marker='^', mfc='b', mec='b', label='INm-')
    plt.plot(t, means['PAGaff'][ind], color='r', marker='^', mfc='r', mec='r', label='PAG')
    plt.plot(t, means['EUSmn'][ind], color='m', marker='^', mfc='m', mec='m', label='EUS Afferent')
    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()

    if savefig:
        if fbmod is not None:
            fig1.savefig('./graphs/Pressure_vol.png',transparent=True)
        fig2.savefig('./graphs/NFR_PGN.png',transparent=True)
        fig3.savefig('./graphs/NFR_PAG.png',transparent=True)

    plt.show()

def plotting_calculator(spike_trains, n_steps, dt, window, index, num, pop):
    # window (ms)
    ind = index[pop]
    n = num[pop]
    fr_conv = np.zeros((n,n_steps))
    
    def moving_avg(x):
        window_size = np.ceil(window/dt).astype(np.int)
        x_cum = np.insert(np.cumsum(x),0,np.zeros(window_size))
        y = (x_cum[window_size:]-x_cum[:-window_size])/(window_size*dt/1000)
        return y

    for gid in range(ind,ind+n):
        spikes = np.zeros(n_steps)
        spiketimes = spike_trains.get_times(gid)
        if len(spiketimes) > 0:
            spikes[(spiketimes/dt).astype(np.int)] = 1
        fr_conv[gid-ind] = moving_avg(spikes)

    means = np.mean(fr_conv,axis=0)
    stdevs = np.std(fr_conv,axis=0)
    
    return means, stdevs