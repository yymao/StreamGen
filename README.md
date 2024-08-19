# StreamGen

**StreamGen** is a Python-based tool designed to integrate satellite orbits in galactic simulations and classify them as streams, shells, or intact satellites. It calculates various physical quantities such as energy, angular momentum, and rosette angles, which help determine the dynamical state of satellite galaxies within their host galaxy.

## Features

- **Orbit Integration**: Simulates the evolution of satellite orbits around a host galaxy, supporting both forward and backward time integration.
- **Classification**: Automatically classifies satellite orbits into streams, shells, or intact satellites based on computed orbital parameters.
- **Monte Carlo Simulations**: Perform Monte Carlo simulations to estimate distributions of orbital  properties, such as energy and angular momentum.
- **Metrics Calculation**: Computes important metrics including energy (`E`), angular momentum (`L`), deltaPsi (angle between successive apocenters), and rosette angles.

## Installation

To install **StreamGen**, you need Python 3.x. You can clone the repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/your-username/StreamGen.git
cd StreamGen
pip install -r requirements.txt
