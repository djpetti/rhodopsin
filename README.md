# Rhodopsin

A simple management framework for deep learning experiments.

## What is it?

Rhodopsin is a minimalistic utility geared toward deep learning researchers.
It's designed to facilitate experimentation by allowing you to change
hyperparameters on-the-fly without restarting your training code. Furthermore,
Rhodopsin can automatically track statistics about your model, and can produce
quick reports.

Rhodopsin is not specific to any particular machine learning framework or
package.

## How does it work?

Rhodopsin rebinds the SIGINT signal, so when you Ctrl-C your training code, it
takes you into a menu that allows you to view statistics and adjust
hyperparameters. It's as simple as that.

## Installation

Rhodopsin can be installed by cloning the Git repository and running the
installation script:

```
~$ python setup.py install
```

## Usage

Here is a very simple example of using Rhodopsin to control an experiment:

```python
from rhodopsin import experiment


class MyExp(experiment.Experiment):
  """ This is a class for my experiment. """

  def _run_training_iteration(self):
    ...
    Code to run a single training iteration.
    ...

  def _run_testing_iteration(self):
    ...
    Code to run a single testing iteration.
    ...


# The number of training iterations to run before every testing iteration.
testing_iters = 100

exp = MyExp(testing_iters)
# Run the training until manually stopped.
exp.train()
```

### Adjusting Hyperparameters

In order for changing a hyperparameter to have any effect, you must explicitly
check for changes in your training code, and take any appropriate action. Here
is an example of reading the set value of a hyperparameter:

```python
def _run_training_iteration(self):
  params = self.get_params()
  # Get the set value for the learning rate.
  learning_rate = params.get_value("learning_rate")

  ...
  Training code.
  ...
```

Incidentally, `Hyperperparameters` has a useful method called `get_changed()`.
This returns a list of all the parameters that have changed since the last call.

```python
params = self.get_params()
print params.get_changed()
# Prints []

params.add("learning_rate", 0.001)
params.add("momentum", 0.9)

print params.get_changed()
# Prints ['learning_rate', 'momentum']
```

### Custom Hyperparameters

So far, Rhodopsin does not specify any default hyperparameters. Therefore, if
you want to add a hyperparameter, and you want its value to be changeable
on-the-fly, you have to specify it manually:

```python
from rhodopsin import experiment, params


class MyExp(experiment.Experiment):
  ...


# Create custom hyperparameter manager.
my_params = params.HyperParams()

# Add custom parameters with initial values.
my_params.add("learning_rate", 0.001)
my_params.add("momentum", 0.9)

experiment = MyExp(100, hyperparams=my_params)
experiment.train()
```

You will now be able to access the values of `learning_rate` and
`momentum` as described above.

### Custom Status

Rhodopsin allows you to view the current and historical values of certain model
statistics or status parameters. By default, the only status parameter is the
number of iterations that have been run. Custom status parameters can be
specified in a similar way to custom hyperparameters:

```python
from rhodopsin import experiment, params


class MyExp(experiment.Experiment):
  ...


# Create custom status parameter manager.
my_status = params.Status()

# Add custom parameters with initial values.
my_status.add("loss", 0.0)
my_status.add("acc", 0.0)

experiment = MyExp(100, status=my_status)
experiment.train()
```

The status parameters can then be updated from within your training code:

```python
def _run_training_iteration(self):
  status = self.get_status()

  ...
  Training code.
  ...

  # Update the values of status parameters.
  status.update("loss", loss)
  status.update("acc", accuracy)
```
