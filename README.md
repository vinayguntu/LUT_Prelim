--- 
# Micturition BMTK README 
---

This version of the Nair Lab BMTK model for the Rat LUT was developed by Erin Shappell (NSF NeuroREU Summer 2019). 

## Primary Contributions
* All (known) sources for values and connections have been added via inline comments.
* Hoc templates have been added (but are not complete due to a lack of data on some values) for the following neurons using data from Vinay's document _LUT-extra.docx_:
  * Hypogastric
  * IMG
  * IND
  * MPG
  * PGN
* A shell script (run.sh) has been added to make running the simulation much easier. Simply type the following line to run all Python scripts needed for the simulation:

```bash

$ ./run.sh

```

* Python script generate_input.py now contains code for the following: (moved to run_bionet.py)
  * Calculating bladder volume over a time period of 10,000 ms
  * Calculating bladder pressure over a time period of 10,000 ms using Grill's polynomial fit equation (based on firing rate of one PGN cell calculated after a prior simulation) (Grill, et al. 2016)
  * Calculating bladder afferent firing rate using the pressure values found in point #2 using Grill's polynomial fit equation (Grill, et al. 2016)
  * Plotting bladder volume, pressure, and afferent firing rate over time (plot trends match up with those found in Grill's 2016 paper)

* Python script plot_test.py now contains code for the following:
  * Calculating and saving "instantaneous" PGN neuron spike rate in PGN_freqs.csv (also contains code for averaged spike rate in intervals of 1000 ms--it has been commented out but was left in the code in case it is useful in the future)

* Synaptic depression (and facilitation) capabilities have now been added to this project (STSP code was provided by Tyler Banks)
  * Synapse file was added: /biophys-components/synaptic-models/stsp.json
  * Mod file was added and compiled: /biophys-components/mechanisms/modfiles/exp2syn-stsp.mod
  * Tyler's script for instantiating the synapse data into our model was added /synapses.py and code for using the script was added to run_bionet.py:

```python

import synapses
synapses.load()

```

* Bladder afferent <<>> PGN feedback loop is now closed
  * Used sample code provided by Kael Dai, Allen Institute 2019 to create the code for this
  * After each block in simulation, the firing rate of the MPG is recorded and used to calculate the firing rate for the bladder afferent using Grill's equations for pressure and bladder afferent firing rate (Grill, et al. 2016)

* Work in Progress: Implementing Guarding Reflex
  * Guarding reflex: the EUS muscles will contract in response to spikes in bladder pressure
  * Implementing this using the feedback loop process used in the bladder afferent <<>> PGN feedback loop
  * Idea is to use pressure to determine whether the EUS motor neuron will have brief increases in an already contracting spiking rate




