from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.simulator.bionet.io_tools import io
from neuron import h
from bmtk.simulator import bionet
from bmtk.simulator.bionet.default_setters.cell_models import loadHOC

import logging
import numpy as np

# Import the synaptic depression/facilitation model
import synapses
synapses.load()

bionet.pyfunction_cache.add_cell_model(loadHOC, directive='hoc', model_type='biophysical')

# Huge thank you to Kael Dai, Allen Institute 2019 for the template code
# we used to create the feedback loop below
class FeedbackLoop(SimulatorMod):
    def __init__(self):
        self._synapses = {}
        self._netcon = None
        self._spike_events = None

        self._high_level_neurons = []
        self._low_level_neurons = []
        self._eus_neurons = []

        self._synapses = {}
        self._netcons = {}

        self._spike_records = {}
        self._glob_press = 0
        self._prev_glob_press = 0
        self._vect_stim = None
        self._spikes = None
        self.times = []
        self.b_vols = []
        self.b_pres = []
        self.press_thres = 17 # cm H20 #40
                 # Lingala, et al. 2016
        self.change_thres = 10 # cm H20 #10
                 # Need biological value for this

    def _set_spike_detector(self, sim):
        for gid in self._low_level_neurons:
            tvec = sim._spikes[gid]
            self._spike_records[gid] = tvec

    def _activate_hln(self, sim, block_interval, firing_rate):
        block_length = sim.nsteps_block*sim.dt/1000.0
        next_block_tstart = (block_interval[1] + 1) * sim.dt/1000.0  # The next time-step
        next_block_tstop = next_block_tstart + block_length  # The time-step when the next block ends

        # This is where you can use the firing-rate of the low-level neurons to generate a set of spike times for the
        # next block
        if firing_rate != 0.0:
            psg = PoissonSpikeGenerator()
            psg.add(node_ids=[0], firing_rate=firing_rate, times=(next_block_tstart, next_block_tstop)) # sec
            self._spike_events = psg.get_times(0)*1000 # convert sec to ms
            n_spikes = len(self._spike_events)
            io.log_info('     _activate_hln firing rate: {:.2f} Hz'.format(n_spikes/block_length))
            if n_spikes > 0:
                # Update firing rate of bladder afferent neurons
                for gid in self._high_level_neurons:
                    nc = self._netcons[gid]
                    for t in self._spike_events:
                        nc.event(t)
                io.log_info('Last spike: {:.1f} ms'.format(self._spike_events[-1]))
            ## Use inhomogeneous input
            #n = len(self._high_level_neurons)
            #psg.add(node_ids=list(range(n)), firing_rate=firing_rate, times=(next_block_tstart, next_block_tstop))            
            #n_spikes = np.zeros(n)
            #for i, gid in enumerate(self._high_level_neurons):
            #    self._spike_events = psg.get_times(i)*1000
            #    n_spikes[i] = len(self._spike_events)
            #    nc = self._netcons[gid]
            #    for t in self._spike_events:
            #            nc.event(t)
            #io.log_info('     _activate_hln firing rate: '+','.join(["%.2f" % (ns/block_length) for ns in n_spikes])+' Hz')
        else:
            io.log_info('     _activate_hln firing rate: 0')

        # If pressure is maxxed, update firing rate of EUS motor neurons 
        # Guarding reflex
        # press_change = self._prev_glob_press - self._glob_press
        # if self._glob_press > press_thres or press_change > change_thres:
            # psg = PoissonSpikeGenerator()
            # eus_fr = self._glob_press*10 + press_change*10 # Assumption: guarding reflex
                                                           # # depends on current pressure
                                                           # # and change from last pressure
            # psg.add(node_ids=[0], firing_rate=eus_fr, times=(next_block_tstart, next_block_tstop))
            # self._spike_events = psg.get_times(0)
            # for gid in self._eus_neurons:
                # nc = self._netcons[gid]
                # for t in self._spike_events:
                    # nc.event(t)
################ Activate higher order based on pressure threshold ##############################

        # if block_interval[1] % 2000 == 1000:  # For fast testing, only add events to every other block
        # if False:  # For testing
        # if self._glob_press > self.press_thres:
        #     io.log_info('      updating pag input')
        #     psg = PoissonSpikeGenerator()
        #     print(self.press_thres)
        #
        #     pag_fr = self.press_thres #change
        #     psg.add(node_ids=[0], firing_rate=pag_fr, times=(next_block_tstart/1000.0, next_block_tstop/1000.0))
        #     if psg.n_spikes() <= 0:
        #         io.log_info('     no psg spikes generated by Poisson distritubtion')
        #     self._spike_events = psg.get_times(0)
        #     for gid in self._pag_neurons:
        #         nc = self._netcons[gid]
        #         for t in self._spike_events:
        #             nc.event(t)
################ Activate higher order based on afferent firing rate ##############################					
        if firing_rate > 10:
            psg = PoissonSpikeGenerator()
            pag_fr = 15
            psg.add(node_ids=[0], firing_rate=pag_fr, times=(next_block_tstart, next_block_tstop))
            self._spike_events = psg.get_times(0)*1000
            n_spikes = len(self._spike_events)
            io.log_info('     pag firing rate: {:.2f} Hz'.format(n_spikes/block_length))
            io.log_info('Last spike: {:.1f} ms'.format(self._spike_events[-1]))
            for gid in self._pag_neurons:
                nc = self._netcons[gid]
                for t in self._spike_events:
                    nc.event(t)
            ## Use inhomogeneous input
            #n = len(self._pag_neurons)
            #psg.add(node_ids=list(range(n)), firing_rate=pag_fr, times=(next_block_tstart, next_block_tstop))            
            #n_spikes = np.zeros(n)
            #for i, gid in enumerate(self._pag_neurons):
            #    self._spike_events = psg.get_times(i)*1000
            #    n_spikes[i] = len(self._spike_events)
            #    nc = self._netcons[gid]
            #    for t in self._spike_events:
            #            nc.event(t)
            #io.log_info('     pag firing rate: '+','.join(["%.2f" % (ns/block_length) for ns in n_spikes])+' Hz')

        io.log_info('\n')

    def initialize(self, sim):
        #####  Make sure to save spikes vector and vector stim object
        # Attach a NetCon/synapse on the high-level neuron(s) soma. We can use the NetCon.event(time) method to send
        # a spike to the synapse. Which, is a spike occurs, the high-level neuron will inhibit the low-level neuron.
        self._spikes = h.Vector(np.array([]))  # start off with empty input
        vec_stim = h.VecStim()
        vec_stim.play(self._spikes)
        self._vect_stim = vec_stim

        self._high_level_neurons = list(sim.net.get_node_set('high_level_neurons').gids())
        self._pag_neurons = list(sim.net.get_node_set('pag_neurons').gids())

        io.log_info('Found {} high level neurons'.format(len(self._high_level_neurons)))
        for gid in self._high_level_neurons:
            cell = sim.net.get_cell_gid(gid)

            # Create synapse
            # These values will determine how the high-level neuron behaves with the input
            new_syn = h.Exp2Syn(0.5, cell.hobj.soma[0])
            new_syn.e = 0.0
            new_syn.tau1 = 0.1
            new_syn.tau2 = 0.3
            self._synapses[gid] = new_syn

            nc = h.NetCon(self._vect_stim, new_syn)
            nc.threshold = sim.net.spike_threshold
            nc.weight[0] = 0.5
            nc.delay = 1.0
            self._netcons[gid] = nc

        io.log_info('Found {} PAG neurons'.format(len(self._pag_neurons)))
        for gid in self._pag_neurons:
            trg_cell = sim.net.get_cell_gid(gid)  # network._rank_node_ids['LUT'][51]

            syn = h.Exp2Syn(0.5, sec=trg_cell.hobj.soma[0])
            syn.e = 0.0
            syn.tau1 = 0.1
            syn.tau2 = 0.3
            self._synapses[gid] = syn

            nc = h.NetCon(self._vect_stim, syn)
            nc.threshold = sim.net.spike_threshold
            nc.weight[0] = 0.5
            nc.delay = 1.0
            self._netcons[gid] = nc

        # Attach another netcon to the low-level neuron(s) that will record
        self._low_level_neurons = list(sim.net.get_node_set('low_level_neurons').gids())
        io.log_info('Found {} low level neurons'.format(len(self._low_level_neurons)))

        self._set_spike_detector(sim)

    def step(self, sim, tstep):
        pass

    def block(self, sim, block_interval):
        """This function is called every n steps during the simulation, as set in the config.json file (run/nsteps_block).

        We can use this to get the firing rate of PGN during the last block and use it to calculate
        firing rate for bladder afferent neuron
        """

        # Calculate the avg number of spikes per neuron
        # return
        block_length = sim.nsteps_block*sim.dt/1000.0  #  time length of previous block of simulation TODO: precalcualte /100
        n_gids = 0
        n_spikes = 0
        for gid, tvec in self._spike_records.items():
            n_gids += 1
            n_spikes += len(list(tvec))  # use tvec generator. Calling this deletes the values in tvec

        avg_spikes = n_spikes/(float(n_gids)*1.0)

        # Calculate the firing rate the the low-level neuron(s)
        fr = avg_spikes/float(block_length)
    
        # Grill function for polynomial fit according to PGN firing rate
	    # Grill, et al. 2016
        def pgn_fire_rate(x):
            f = 2.0E-03*x**3 - 3.3E-02*x**2 + 1.8*x - 0.5
            f = max(f,0.0)
            # Take absolute value of firing rate if negative
            # if f < 0:
            #     f *= -1

            return f

        # Grill function for polynomial fit according to bladder volume
	    # Grill, et al. 2016
        def blad_vol(vol):
            f = 1.5*20*vol - 10 #1.5*20*vol-10

            return f

        # Grill function returning pressure in units of cm H20
	    # Grill, et al. 2016
        def pressure(fr,v):
            p = 3.0*fr + 0.5*v

            # Round negative pressure up to 0
            if p < 0:
                p = 0

            return p 

        # Grill function returning bladder afferent firing rate in units of Hz
	    # Grill, et al. 2016
        def blad_aff_fr(p):
            fr1 = -3.0E-08*p**5 + 1.0E-5*p**4 - 1.5E-03*p**3 + 7.9E-02*p**2 - 0.6*p
            fr1 = max(fr1,0.0)
            # Take hard max or absolute value of negative firing rates
            # if fr1 < 0:
            #     fr1 *= -1#fr1 *= 0

            return fr1 # Using scaling factor of 5 here to get the correct firing rate range

        # Calculate bladder volume using Grill's polynomial fit equation
        v_init = 0.05       # TODO: get biological value for initial bladder volume
        fill = 0.05 	 	# ml/min (Asselt et al. 2017)
        fill /= (1000 * 60) # Scale from ml/min to ml/ms
        void = 4.6 	 		# ml/min (Streng et al. 2002)
        void /= (1000 * 60) # Scale from ml/min to ml/ms
        max_v = 1.5 		# ml (Grill et al. 2019) #0.76
        vol = v_init

        # Update blad aff firing rate
        t = sim.h.t-block_length*1000.0
        if fr > 0:
            PGN_fr = pgn_fire_rate(fr)

            # Filling: 0 - 7000 ms
            # if t < 7000 and vol < max_v:
		# vol = fill*t*150 + v_init
			
			# Filling: 0 - 54000 ms
            if t < 60000 and vol < max_v:
                vol = fill*t*20 + v_init
           
			# # Voiding: 7000 - 10,000 ms
            # else:
                # vol = max_v - void*(10000-t)*100


		   # Voiding: 54000 - 60000 ms
            # else:
                # vol = max_v - void*(60000-t)*100

            # Maintain minimum volume
            if vol < v_init:
                vol = v_init
            grill_vol = blad_vol(vol)

            # Calculate pressure using Grill equation
            p = pressure(PGN_fr, grill_vol)

            # Update global pressure (to be used to determine if EUS motor
            # needs to be updated for guarding reflex)
            self._prev_glob_press = self._glob_press
            self._glob_press = p 

            # Calculate bladder afferent firing rate using Grill equation
            bladaff_fr = blad_aff_fr(p)

            io.log_info('PGN firing rate = %.2f Hz' %fr)
            io.log_info('Volume = %.2f ml' %vol)
            io.log_info('Pressure = %.2f cm H20' %p)
            io.log_info('Bladder afferent firing rate = {:.2f} Hz'.format(bladaff_fr))

            # Save values in appropriate lists
            self.times.append(t)
            self.b_vols.append(vol)
            self.b_pres.append(p)
			# b_aff.append(bladaff_fr)
			# pgn_fir.append(fr)

            # Set the activity of high-level neuron
            self._activate_hln(sim, block_interval, bladaff_fr)

        else:
            self._activate_hln(sim, block_interval, fr) 

        # NEURON requires resetting NetCon.record() every time we read the tvec.
        self._set_spike_detector(sim)



    def finalize(self, sim):
        pass
