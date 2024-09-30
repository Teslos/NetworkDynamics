# Introduction
We are trying to get NetworkDynamics to run the general nonlinear oscillators.
By creating the graph we can create different enviroments and train different
cases. 
# Examples
**Duffing.jl** - calculates the Duffing oscillators on the network graph

**FitzHug-Nagumo** - calculates the FitzHugh-Nagumo model for the nodes in the network graph.

**Diffusion-optimiz.jl** - Optimization of the network dynamics for the diffusion constant.

# How to use

# Results
The oscillations in Duffing network using the random eigen frequencies chosen randomly around given value $\omega$:


![Duffing oscillators](./figs/duffing_barabasi_albert.png)

Using the larger deviation from the average frequency produces this result:

![Duffing oscillator 0.5](./figs/duffing_barabasi_albert_0_5.png)

This are solution to the network graph of the brain system:

![Duffing oscillator brain](./figs/duffing_brain_graph.png)

# Learning using reservoir computing
The learning procedure for the cases of Drybean classification uses the reservoir compute by network of FHN oscillators.
The network state is set between equilibrium and chaotic state. We expect highest computing ability in this particular state. 
Additional things to be done and tested are:
- Training efficiency with different network topologies
- Parallel solution of the reservoir using the GPU and maybe ModelingToolkit library instead of NetworkDynamics
- Interactive demo to set up the active nodes in the graph (using JavaScript)

# References

Chaos 31, 013108 (2021)