# Yakovlev-BS-Thesis
### Selection of concordant models with complexity control 

The paper investigates the problem of choosing the architecture of a deep learning model. A model is understood as a superposition of functions differentiable by parameters. The problem of choosing a model of a special type in which the structural parameters of the model are a function of the input object to be analyzed. To increase the interpretability of the final architecture, it also depends on the parameter of the required complexity, which allows us to find a compromise between the complexity of the model and its predictive characteristics during operation. The complexity of the model is understood as the minimum length of the description,
the minimum amount of information required to transmit information about the model and the dataset.

### Basic experiment
The purpose of the experiment is to search for architecture in the case of two modalities:
ordinary MNIST and MNIST where each picture is transposed. To run the experiment, write the 
following in the terminal:
```console
python search.py --config mnist_basic.cfg
```

To visualize obtained cells, run the following:
```console
python visualize.py
```
