# dominating_sets_patternboost
## Patternboost for Dominating Set Sequences of Graphs and Trees

This repository contains the code for PatternBoost, adapted to find graphs and trees with non-log-concave dominating set sequences.  Patternboost was created in https://github.com/zawagner22/transformers_math_experiments written by Charton-Ellenberg-Wagner-Williamson and then adapted to find trees with non-log-concave independent set sequences in https://github.com/ericgramos/TreeUnimodalityPatternBoost written by Ramos-Sun.

The Patternboost algorithm alternates between a greedy local search and transformer-based reinforcement learning loop to find new examples in combinatorics.

For more detail on Patternboost itself, see e.g. https://github.com/zawagner22/transformers_math_experiments

## Installation
Prerequisites
* Python 3.13.9 (with standard packages)
* Julia Version 1.11.7
* NVIDIA GPU with properly installed CUDA drivers

We found that the Patternboost code sometimes did not function correctly with the latest versions of Python and Julia.  So, we would recommend using the exact versions specified above.  First, install Julia, e.g. by visiting https://julialang.org/downloads/  , or by executing in the command line something like: 

`curl -fsSL https://install.julialang.org | sh`

The Julia code in Patternboost might use some Julia packages.  To install several such packages at once, you can use a command such as the following.

`julia -e 'import Pkg; Pkg.add(["ArgParse", "Random", "LinearAlgebra", "Statistics", "StatsBase", "Dictionaries", "Printf", "Plots", "Combinatorics", "Dates", "JSON", "Polynomials","DataStructures"])'`

Once Patternboost is installed, it can be run from a command line with a command such as the following.

`python fc_loop.py --max_epochs 10 --sample-only 100000 --max-steps 8000 --n_tokens 60 --dump_path experiment_output --exp_name test_run --nb_threads 64 --nb_local_searches 64000`

You can change the problem by changing the relevant line in `search_fc.jl` to call a file of the form `problem_(problem_name).jl`

## Troubleshooting

Patternboost will typically start to write files from its experiment after 5 or 10 minutes.  If you find that it is taking over an hour to write an files or show any output in the command line, something is probably wrong.  Also, a very common error is that a file such as `search_output_txt_1` does not exist.  This error occurs when there is a problem in the program such that the first output file is not written, so the program cannot proceed further (since the output file is empty).

## Setup
Clone the repository:
git clone https://github.com/sheilman77/dominating_sets_patternboost

## Usage

## Contributing
Feel free to explore other problems or propose extensions to the PatternBoost algorithm!
