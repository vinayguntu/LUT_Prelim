# Updates 4/16
* Very interconnected - PNG calculation depends on PGN [here](feedback_loop.py)
* Need to figure out why can't shrink Hz values for EUS and PAG in generate file
* Overall model is functioning
    * Need to figure out how to make the model fire at 15 Hz
* Need to plot one membrane voltage

Try:
  - generate MPG input with 1 Hz instead of PAG
    * Need to generate it in the generate_input.py file
    * Need to remove PAG spikes
    * Switch low level neuron on [simulation_config](jsons/simulation_config.json)
    * Adapt synaptic time constant

Problem Definition:
* Biological:
    * When 10 Hz signal is applied there is still a high value in the bladder pressure
* Computational:
    * Very interconnected
    * 10 Hz not generating the bladder drop
    * Cannot easily remove low-level neurons