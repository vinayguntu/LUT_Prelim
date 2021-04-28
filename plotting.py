import matplotlib.pyplot as plt
import numpy as np

def plot_figure(ba_means, ba_stdevs, pgn_means, pgn_stdevs, eus_means, eus_stdevs,
                inmm_means, inmm_stdevs, inmp_means, inmp_stdevs, fbmod, savefig=True):

    # Only plot one point each 1000 samples
    plt_ba_means = []
    plt_ba_stdevs = []
    for n in np.arange(0,len(ba_means), 1000):
        plt_ba_means.append(ba_means[n])   
        plt_ba_stdevs.append(ba_stdevs[n]) 

    # Only plot one point each 1000 samples
    plt_pgn_means = []
    plt_pgn_stdevs = []
    for n in np.arange(0,len(pgn_means),1000):
        plt_pgn_means.append(pgn_means[n])   
        plt_pgn_stdevs.append(pgn_stdevs[n]) 

    # Only plot one point each 1000 samples
    plt_eus_means = []
    plt_eus_stdevs = []
    for n in np.arange(0,len(eus_means),1000):
        plt_eus_means.append(eus_means[n])   
        plt_eus_stdevs.append(eus_stdevs[n]) 

    # Only plot one point each 1000 samples
    plt_inmm_means = []
    plt_inmm_stdevs = []
    for n in np.arange(0,len(inmm_means), 1000):
        plt_inmm_means.append(inmm_means[n])   
        plt_inmm_stdevs.append(inmm_stdevs[n]) 

    # Only plot one point each 1000 samples
    plt_inmp_means = []
    plt_inmp_stdevs = []
    for n in np.arange(0,len(inmp_means), 1000):
        plt_inmp_means.append(inmp_means[n])   
        plt_inmp_stdevs.append(inmp_stdevs[n]) 

    fig0 = plt.figure()
    plt.plot(np.arange(0,len(ba_means)/10,100), plt_ba_means, 
                 color='b', marker='^', mfc='b', mec='b', label='Bladder Afferent')
    plt.xlabel('Time (t) [ms]')

    plt.plot(np.arange(0,len(pgn_means)/10,100), plt_pgn_means,  
                 color='g', marker='o', mfc='g', mec='g', label='PGN')

    plt.plot(np.arange(0,len(eus_means)/10,100), plt_eus_means, 
                 color='k', marker='D', mfc='k', mec='k', label='EUS Motor Neurons')
    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()

    #Plot bladder volume and bladder pressure
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

    fig2 = plt.figure()

    plt.plot(np.arange(0,len(inmm_means)/10,100), plt_inmm_means, 
                color='b', marker='^', mfc='b', mec='b', label='INm-')
    plt.xlabel('Time (t) [ms]')

    plt.plot(np.arange(0,len(inmp_means)/10,100), plt_inmp_means, 
                 color='r', marker='^', mfc='r', mec='r', label='PAG')
    plt.xlabel('Time (t) [ms]')

    plt.plot(np.arange(0,len(eus_means)/10,100), plt_eus_means, 
                 color='m', marker='^', mfc='m', mec='m', label='EUS Afferent')
    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()

    if savefig:
        fig0.savefig('./graphs/NFR_PGN.png',transparent=True)
        fig1.savefig('./graphs/Pressure_vol.png',transparent=True)
        fig0.savefig('./graphs/NFR_PAG.png',transparent=True)

    plt.show()

def plotting_calculator(spike_trains, n_steps, dt, window_size, arange1, arange2, arange3=0, multiplier=1):
    # Plot PGN firing rate
    means = np.zeros(n_steps)
    fr_conv = np.zeros((arange2,n_steps))
    window = np.ones(window_size)

    for gid in np.arange(arange1, arange3 + arange2):
        spikes = np.zeros(n_steps)
        spiketimes = spike_trains.get_times(gid)
        if len(spiketimes) > 0:
            spikes[(spiketimes/dt).astype(np.int)] = 1.0

        frs = np.convolve(spikes, window)[:n_steps]
        means += frs
        fr_conv[gid-arange1] = frs

        means /= arange2*multiplier
        stdevs = np.std(fr_conv,axis=0)
    
    return means, stdevs