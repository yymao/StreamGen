# StreamGen

This repository contains the tool introduced in [Dropulic et al. 2024](), "StreamGen: Connecting Semi-Analytic Tidal Debris to Dark Matter Halos." **StreamGen** is a Python-based a tool which estimates the morphology of semi-analytic tidal debris as belonging to one of three categories: intact, stream-like, and shell-like. StreamGen is built for the semi-analytic satellite galaxy generator [SatGen](https://academic.oup.com/mnras/article/502/1/621/6066532), but in principle is applicable to any host-satellite pair given orbital quantities. The morphology metric is built upon the work of [Hendel and Johnston 2015](https://academic.oup.com/mnras/article/454/3/2472/1194235). 

The `streamgen_starter.ipynb` provides a starting point for working with StreamGen. It loads in an example SatGen merger tree and walks the user through to substructure morphology prediction. Note to the reader: StreamGen implements multiprocessing to sample over orbits of particles within a satellite; this step will go faster if you use this fully. We are also excited to explore an implementation using [jax](https://jax.readthedocs.io/en/latest/) in the future. This first version of StreamGen will catch many errors that might arise from extracting necessary quantities from the merger trees to substructure catalog construction, but it is possible that the user may encounter new errors. Please feel free to suggest changes to the script and contact Adriana Dropulic (dropulic [at] princeton [dot] edu).

## Dependencies 

[`SatGen`](https://github.com/shergreen/SatGen) is required to run StreamGen. However, it would be possible for the user to modify `load_galaxy.py` to load in alternative merger trees. The user could alternatively take the core functions from StreamGen and apply them to any simulation. If you use StreamGen in part or in full, please cite [Dropulic et al. 2024](). 

All other dependencies are included in `setup.py`. 

## Installation

To install **StreamGen**, the authors recommend working in a new conda environment with Python version >= 3.8. First, create a new directory, into which you can clone SatGen and StreamGen as separate directories. 

You can clone the StreamGen repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/your-username/StreamGen.git
cd StreamGen
pip install .
```

If you receive an error using HTTPS, try using an SSH key. This error could arise due to the size of the example SatGen merger tree (68M). 

Once SatGen and StreamGen, as well as their dependencies are installed, the authors recommend including the path to SatGen at the beginning of the StreamGen `.py` files so they can call SatGen functions. Please contact Adriana Dropulic (dropulic [at] princeton [dot] edu) with any inquiries.


