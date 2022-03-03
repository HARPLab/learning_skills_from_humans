# Actively Learning to Prepare a Human's Preferred Meal Configuration

Maintainer: Pat Callaghan\
Last updated: March 3rd, 2022

## Expected outputs

Upon executing the main program, one should see terminal outputs indicating successful
instantiation of the following python objects:

- An ideal pizza
- A particle filter
- A set of particles
- A query generator

Subsequent terminal outputs should display the expected parameter values and importance
weights of the learned reward function after each (automated) query. One will also
see terminal outputs indicating when the particle filter resamples its set of particles.

At the end of a successful program execution, one will see two plots. The first is
a hypothesized pizza generated according to the learned reward model as it is at
the **end** of execution. The second plot is a hypothesized pizza generated according
to the reward model when it was most similar to the **true** model at any point
during execution.

## Running the code

### Main program

From within the top level of the learning\_humans\_preferred\_meal\_prep,
execute the following from the commandline:

```bash
./robot_chef_program --<flag_1> <arg_1> ... --<flag_n> <arg_n>
```

Please find the arguments and their corresponding flags in the
utils/arguments.py file.

### Data Visualization

Upon the program's successful execution, one can locate the corresponding data
in the pertinent csv files found within the data/ subdirectory.

To visualize the results, cd into the utils/ subdirectory and execute the
following from the commandline:

```bash
./create_data_plots.py ../data/<pertinent subdirectory>/<pertinent file>.csv \
 --context <desired plot context>
```

with current eligible contexts:

- expected\_values
- choice\_comparisons
