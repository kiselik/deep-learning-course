# deep-learning-course

## Installation
System requrements:
  - Python 3.6, 
  - python-mnist package
  - numpy package

To simplify the obtaining dataset, please use get_training_data script. 
```
pip3 install python-mnist numpy
python3 get_training_data.py
```
## Getting started
neural_network.py implemented NeuralNetwork  class and related functionality. 
Run example:
```
python3 neural_network.py train-images/ 10 0.05 0.01 100
```
Inputs:
 - folder with training data (in .gz format)
 - stop criterion by the number of eras
 - stop criterion for minimizing cross-entropy
 - learning Speed
 - the number of neurons in the hidden layer
