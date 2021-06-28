![Using Genetic Algorithm  for feature selection in text dataset](https://www.dropbox.com/s/htjado9iveuqr6d/Picturea.png?raw=1)

# Using Genetic Algorithm  for feature selection in text dataset  
This project presents a wrapper approach of feature selection using a genetic algorithm. The multinomial event model of Naïve Bayes was used as a fitness function to determine the selected features of the documents. Different experiments have been performed on the 20newsgroups dataset to see the impact of the population size and the number of generations. The experiments were compared with the classification without feature selection using different evaluation metrics. All the experiment results showed that feature selection using the genetic algorithm will positively affect the classification performance.

# The results:
![four expermints comparsion](https://www.dropbox.com/s/9wzwif2744ulz0q/Picture1.png?raw=1)
![four expermints results comparsion](https://www.dropbox.com/s/9wzwif2744ulz0q/Picture5.png?raw=1)
### Expermint A
![Expermint A](https://www.dropbox.com/s/chkuhumf422eq69/Picture11.png?raw=1)
### Expermint B
![Expermint B](https://www.dropbox.com/s/5nwk5h1wgiggprn/Picture22.png?raw=1)
### Expermint C
![Expermint C](https://www.dropbox.com/s/k9sqs5u55jk7j95/Picture33.png?raw=1)
### Expermint D
![Expermint D](https://www.dropbox.com/s/304ax1zjuvzgwxy/Picture4.png?raw=1)

---- 
# Below some information about the algorithm and its library (PyGAD)

## PyGAD:  Genetic Algorithm in Python

[PyGAD](https://pypi.org/project/pygad) is an open-source easy-to-use Python 3 library for building the genetic algorithm and optimizing machine learning algorithms. It supports Keras and PyTorch.

Check documentation of the [PyGAD](https://pygad.readthedocs.io/en/latest).

[![Downloads](https://pepy.tech/badge/pygad)](https://pepy.tech/project/pygad) ![Docs](https://readthedocs.org/projects/pygad/badge)

![PYGAD-LOGO](https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png)

[PyGAD](https://pypi.org/project/pygad) supports different types of crossover, mutation, and parent selection. [PyGAD](https://pypi.org/project/pygad) allows different types of problems to be optimized using the genetic algorithm by customizing the fitness function. 

The library is under active development and more features are added regularly. If you want a feature to be supported, please check the **Contact Us** section to send a request.


# Installation

To install [PyGAD](https://pypi.org/project/pygad), simply use pip to download and install the library from [PyPI](https://pypi.org/project/pygad) (Python Package Index). The library lives a PyPI at this page https://pypi.org/project/pygad.

Install PyGAD with the following command:

```python
pip install pygad
```

PyGAD is developed in Python 3.7.3 and depends on NumPy for creating and manipulating arrays and Matplotlib for creating figures. The exact NumPy version used in developing PyGAD is 1.16.4. For Matplotlib, the version is 3.1.0.

To get started with PyGAD, please read the documentation at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Source Code

The source code of the PyGAD' modules is found in the following GitHub projects:

- [pygad](https://github.com/ahmedfgad/GeneticAlgorithmPython): (https://github.com/ahmedfgad/GeneticAlgorithmPython)
- [pygad.nn](https://github.com/ahmedfgad/NumPyANN): https://github.com/ahmedfgad/NumPyANN
- [pygad.gann](https://github.com/ahmedfgad/NeuralGenetic): https://github.com/ahmedfgad/NeuralGenetic
- [pygad.cnn](https://github.com/ahmedfgad/NumPyCNN): https://github.com/ahmedfgad/NumPyCNN
- [pygad.gacnn](https://github.com/ahmedfgad/CNNGenetic): https://github.com/ahmedfgad/CNNGenetic
- [pygad.kerasga](https://github.com/ahmedfgad/KerasGA): https://github.com/ahmedfgad/KerasGA
- [pygad.torchga](https://github.com/ahmedfgad/TorchGA): https://github.com/ahmedfgad/TorchGA

The documentation of PyGAD is available at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Documentation

The documentation of the PyGAD library is available at [Read The Docs](https://pygad.readthedocs.io) at this link: https://pygad.readthedocs.io. It discusses the modules supported by PyGAD, all its classes, methods, attribute, and functions. For each module, a number of examples are given.

If there is an issue using PyGAD, feel free to post at issue in this [GitHub repository](https://github.com/ahmedfgad/GeneticAlgorithmPython) https://github.com/ahmedfgad/GeneticAlgorithmPython or by sending an e-mail to ahmed.f.gad@gmail.com. 

If you built a project that uses PyGAD, then please drop an e-mail to ahmed.f.gad@gmail.com with the following information so that your project is included in the documentation.

- Project title
- Brief description
- Preferably, a link that directs the readers to your project

Please check the **Contact Us** section for more contact details.

# Life Cycle of PyGAD

The next figure lists the different stages in the lifecycle of an instance of the `pygad.GA` class. Note that PyGAD stops when either all generations are completed or when the function passed to the `on_generation` parameter returns the string `stop`.

![PyGAD Lifecycle](https://user-images.githubusercontent.com/16560492/89446279-9c6f8380-d754-11ea-83fd-a60ea2f53b85.jpg)

The next code implements all the callback functions to trace the execution of the genetic algorithm. Each callback function prints its name.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

fitness_function = fitness_func

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop)

ga_instance.run()
```

Based on the used 3 generations as assigned to the `num_generations` argument, here is the output.

```
on_start()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_stop()
```
