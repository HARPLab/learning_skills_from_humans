# Actively Learning Skills to Prepare a Human's Preferred Meal

Maintainer: Pat Callaghan\
Last updated: March 11th, 2022

## Expected outputs

Upon executing the main program, one should see terminal outputs indicating the human's
desired parameters and weights. Next, one should see indications of the following
Python objects' successful instantiation:

- An ideal pizza
- A particle filter
- A set of particles
- A query generator

Subsequent terminal outputs should display the expected parameter values and importance
weights of the learned reward function after each (simulated) query. One will also
see terminal outputs indicating when the particle filter resamples its set of particles.

At the end of a successful program execution, one should see a plot of a hypothesized
pizza generated according to the learned reward model when it was most similar to
the **true** model at **any** point during execution. Depending on the feasibility
of the learned model's parameters and the error threshold one specifies for generating
a pizza, generation can take several minutes.

## Running the code

### Installation

From the top level of the learning\_skills\_from\_humans directory, execute the
following from the commandline:

```bash
conda env create -f learning_skills_from_humans.yml
```

One can optionally modify the primary module to be executable:

```bash
sudo chmod +x robot_chef_program.py
```

### Main program

From within the top level of the learning\_skills\_from\_humans directory,
execute the following from the commandline:

```bash
python robot_chef_program --<flag_1> <arg_1> ... --<flag_n> <arg_n>
```

And if one opted to make the primary module executable:

```bash
./robot_chef_program --<flag_1> <arg_1> ... --<flag_n> <arg_n>
```

Please find the arguments and their corresponding flags in the arguments.py
file located within the utils/ sub-directory.

### Data Visualization

Upon successful program execution, one can locate the corresponding data
in the pertinent csv files found within the data/ sub-directory.

To visualize the results, cd into the utils/ sub-directory and execute the
following from the commandline:

```bash
./create_data_plots.py ../data/<pertinent sub-directory>/<pertinent file>.csv \
 --context <desired plot context>
```

with current eligible contexts:

- expected\_values
- choice\_comparisons
