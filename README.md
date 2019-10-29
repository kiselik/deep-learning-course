# deep-learning-course

## Installation
System requrements:
  - Python 3.6, 
  - python-mnist package
  - numpy package

To simplify the obtaining dataset, please use get_training_data script. 
```
pip3 install python-mnist numpy
python get_training_data.py
```
## Getting started
neural_network.py implemented NeuralNetwork  class and related functionality. 
Run example:
```
python3 neural_network.py train-images/ 30 0.1 300 10 60
```
Inputs:
 - [data folder] folder with training data (in .gz format)
 - [epochs] stop criterion by the number of eras
 - [learn rate]
 - [hidden size] the number of neurons in the hidden layer
 - [output size] the number of neurons in the output layer
 - [batch_size]
