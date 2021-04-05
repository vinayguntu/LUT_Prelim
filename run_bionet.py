import os, sys, logging, faulthandler
from bmtk.simulator import bionet
from bmtk.simulator.bionet.default_setters.cell_models import loadHOC
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils.reports.spike_trains import SpikeTrains
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.simulator.bionet.io_tools import io
import numpy as np
from neuron import h
import pandas as pd
# from bmtk.analyzer.cell_vars import plot_report
from feedback_loop import FeedbackLoop
from plotting import plot_figure, plotting_calculator
# Import the synaptic depression/facilitation model
import synapses
import pickle

"""
Basic Logging features, disable faulthandler if you don't want stacktraces printed
logging determines the level and file to save logs to (might be worth moving location)
"""
faulthandler.enable()
logging.basicConfig(filename='error_logs/debug_run.log', level=logging.DEBUG)

synapses.load()
logging.info('Synapses Loaded')

press_thres = 17 # cm H20 #40
                 # Lingala, et al. 2016
change_thres = 10 # cm H20 #10
                 # Need biological value for this

numBladaff  = 10 
numEUSaff   = 10
numPAGaff   = 10
numIND      = 10
numHypo     = 10
numINmplus  = 10
numINmminus = 10
numPGN      = 10
numFB       = 10
numIMG      = 10
numMPG      = 10
numEUSmn    = 10
numBladmn   = 10

Blad_gids = 0
EUS_gids = Blad_gids + numBladaff
PAG_gids = EUS_gids + numEUSaff
IND_gids = PAG_gids + numPAGaff
Hypo_gids = IND_gids + numIND
INmplus_gids = Hypo_gids + numHypo
INmminus_gids = INmplus_gids + numINmplus
PGN_gids = INmminus_gids + numINmminus
FB_gids = PGN_gids + numPGN
IMG_gids = FB_gids + numFB
MPG_gids = IMG_gids + numIMG
EUSmn_gids = MPG_gids + numMPG
Bladmn_gids = EUSmn_gids + numEUSmn

bionet.pyfunction_cache.add_cell_model(loadHOC, directive='hoc', model_type='biophysical')
logging.info('Cell model added')

def run(config_file):

    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    fbmod = FeedbackLoop()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.add_mod(fbmod)  # Attach the above module to the simulator.
    sim.run()

    spikes_df = pd.read_csv('output/spikes.csv', sep=' ')
    # spike_trains = SpikeTrains.from_sonata('output/spikes.h5')

    print(spikes_df['node_ids'].unique())
    print(spikes_df[(spikes_df['node_ids'] >= 50) & (spikes_df['node_ids'] < 75)])

    plotting_dict = {}
    plotting_dict['n_steps'] = sim.n_steps
    plotting_dict['dt'] = sim.dt
    # plotting_dict['spike_trains'] = spike_trains
    plotting_dict['times'] = fbmod.times
    plotting_dict['b_vols'] = fbmod.b_vols
    plotting_dict['b_pres'] = fbmod.b_pres

    f = open("output/plotting.pkl", "wb")
    pickle.dump(plotting_dict,f)
    f.close()
    
    bionet.nrn.quit_execution()

def build_plots():
    objects = []
    with (open("output/plotting.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    plotting_dict = objects[0]
    
    #plotting
    ba_means, ba_stdevs = plotting_calculator(plotting_dict, 60000, 0, numBladaff, multiplier=2)
    pgn_means, pgn_stdevs = plotting_calculator(plotting_dict, 60000, PGN_gids, numPGN, PGN_gids, multiplier=2)
    eus_means, eus_stdevs = plotting_calculator(plotting_dict, 60000, EUSmn_gids, numEUSmn, EUSmn_gids, multiplier=2)
    inmm_means, inmm_stdevs = plotting_calculator(plotting_dict, 10000, INmminus_gids, numINmminus, INmminus_gids)
    inmp_means, inmp_stdevs = plotting_calculator(plotting_dict, 10000, PAG_gids, numINmplus, PAG_gids)

    plot_figure(ba_means, ba_stdevs, pgn_means, pgn_stdevs, eus_means, eus_stdevs, inmm_means, inmm_stdevs, inmp_means, inmp_stdevs, plotting_dict)


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
        build_plots()
    else:
        run('config.json')
