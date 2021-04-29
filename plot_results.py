import sys
from bmtk.simulator import bionet
from bmtk.utils.reports.spike_trains import SpikeTrains
from feedback_loop import FeedbackLoop
import numpy as np
import pandas as pd
from plotting import plot_figure, plotting_calculator

num = {
'Bladaff' : 10,
'EUSaff'  : 10,
'PAGaff'  : 10,
'IND'     : 10,
'Hypo'    : 10,
'INmplus' : 10,
'INmminus': 10,
'PGN'     : 10,
'FB'      : 10,
'IMG'     : 10,
'MPG'     : 10,
'EUSmn'   : 10,
'Bladmn'  : 10
}
gids = {}
ind = 0
for pop,n in num.items():
    gids[pop] = ind
    ind += n

def run(config_file=None,sim=None):
    if config_file is not None:
        conf = bionet.Config.from_json(config_file, validate=True)
        dt = conf['run']['dt']
        n_steps = np.ceil(conf['run']['tstop']/dt+1).astype(np.int)
        fbmod = None
    if sim is not None:
        n_steps = sim.n_steps
        dt = sim.dt
        fbmod = sim._sim_mods[[isinstance(mod,FeedbackLoop) for mod in sim._sim_mods].index(True)]
    print(n_steps,dt)
    spikes_df = pd.read_csv('output/spikes.csv', sep=' ')
    print(spikes_df['node_ids'].unique())
    spike_trains = SpikeTrains.from_sonata('output/spikes.h5')

    #plotting
    pops = ['Bladaff','PGN','PAGaff','EUSmn','INmminus']
    windows = [200]*3+[1000]*2
    means = {}
    stdevs = {}
    for pop,win in zip(pops,windows):
        means[pop], stdevs[pop] = plotting_calculator(spike_trains, n_steps, dt, win, gids, num, pop)
    
    plot_figure(means, stdevs, n_steps, dt, tstep=100, fbmod=fbmod)


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(config_file=sys.argv[-1])
    else:
        run(config_file='jsons/simulation_config.json')

