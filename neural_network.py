import sys
from datetime import datetime
import numpy as np
from mnist import MNIST
INPUT_SIZE = 784
OUTPUT_SIZE = 10
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
def logistic(arg):
    return 1 / (1 + np.exp(-arg))
def softmax(arg):
    res = np.zeros(arg.shape)
    sumres = 0
    for i, row in enumerate(arg):
        res[i] = np.exp(row)
        sumres += res[i].sum()
    return res/sumres
def deriv(func, arg):
    return func(arg) * (1 - func(arg))
class NeuralNetwork:
    weights_layers = [[], []]
    hidden_layer = np.array([])
    input_layer = np.array([])
    output_layer = np.array([])
    output_layer_expected = np.array([])
    epochs = 100
    cross_entropy_min = 0.05
    learn_rate = 0.01
    hidden_size = 300
    def __init__(self, epochs, cross_entropy, learn_rate, hidden_size):
        self.epochs = epochs
        self.cross_entropy_min = cross_entropy
        self.learn_rate = learn_rate
        self.hidden_size = hidden_size
        self.hidden_layer = np.zeros(hidden_size)
    def reset_weights(self):
        self.weights_layers[0] = 2*np.random.rand(INPUT_SIZE, self.hidden_size) -1
        self.weights_layers[1] = 2*np.random.rand(self.hidden_size, OUTPUT_SIZE) -1
    def __calc_hidden(self):
        self.hidden_layer = logistic(np.dot(self.input_layer, self.weights_layers[0]))
    def __calc_output(self):
        self.__calc_hidden()
        self.output_layer = softmax(np.dot(self.hidden_layer, self.weights_layers[1]))
    def __correct_weights(self):
        gradient_weights = [
            np.zeros((INPUT_SIZE, self.hidden_size)),
            np.zeros((self.hidden_size, OUTPUT_SIZE))
        ]
        delta1 = np.zeros(self.hidden_size)
        delta2 = np.zeros(OUTPUT_SIZE)
        for i in range(self.hidden_size):
            delta2 = self.output_layer - self.output_layer_expected
            gradient_weights[1][i] = np.dot(delta2, self.hidden_layer[i])
        for i in range(self.hidden_size):
            delta1[i] += np.dot(delta2, self.weights_layers[1][i]) * deriv(logistic, self.hidden_layer[i])
        for i in range(INPUT_SIZE):
            gradient_weights[0][i] = np.dot(delta1, self.input_layer[i])
        #correct weights
        for layer in range(1):
            self.weights_layers[layer] -= self.learn_rate * gradient_weights[layer]
    def __set_input(self, input_layer, label):
        self.input_layer = input_layer
        self.output_layer_expected = label
    def train(self, data, labels):
        for epoch in range(self.epochs):
            correct = 0
            data, labels = unison_shuffled_copies(data, labels)
            for i in range(len(data)):
                #if i % 1000 == 1:
                #    print(i, self.output_layer.max(), self.output_layer.argmax(), self.output_layer_expected.argmax())
                self.__set_input(data[i], labels[i])
                self.__calc_output()
                if self.output_layer.argmax() == self.output_layer_expected.argmax():
                    correct += 1
                self.__correct_weights()
            precision = correct / len(data)
            #calc cross entropy
            cross_entropy = 0
            for i in range(len(data)):
                self.__set_input(data[i], labels[i])
                index = self.output_layer_expected.argmax()
                self.__calc_output()
                cross_entropy -= np.log(self.output_layer[index])
            cross_entropy = cross_entropy / len(data)
            print(str(datetime.now()), 'Epoch:', epoch, 'Cross entropy:', cross_entropy, 'Precision:', precision)
            if cross_entropy < self.cross_entropy_min:
                break
    def test(self, data, labels):
        correct = 0
        for i in range(len(data)):
            self.__set_input(data[i], labels[i])
            self.__calc_output()
            if self.output_layer_expected[self.output_layer.argmax()] == 1:
                correct += 1
        return correct / len(data)
def read_mnist_data(data_folder):
    mndata = MNIST(data_folder)
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    np_train_labels = np.zeros((len(train_labels), OUTPUT_SIZE))
    for i in range(len(train_labels)):
        np_train_labels[i][train_labels[i]] = 1
    np_test_labels = np.zeros((len(test_labels), OUTPUT_SIZE))
    for i in range(len(test_labels)):
        np_test_labels[i][test_labels[i]] = 1
    return np.array(train_images)/255, np_train_labels, np.array(test_images)/255, np_test_labels
def main(argv):
    if len(argv) != 6:
        print("""Usage:
python neural_network.py [data folder] [epochs] [max error] [learn rate] [hidden size]""")
        sys.exit()
    else:
        data_folder = argv[1]
        epochs = int(argv[2])
        cross_entropy = float(argv[3])
        learn_rate = float(argv[4])
        hidden_size = int(argv[5])
        #print(data_folder, epochs, cross_entropy, learn_rate, hidden_size)
        print('Loading data from ', data_folder)
        train_images, train_labels, test_images, test_labels = read_mnist_data(data_folder)
        print('Found', len(train_images), 'training images')
        print('Found', len(test_images), 'testing images')
        network = NeuralNetwork(epochs, cross_entropy, learn_rate, hidden_size)
        network.reset_weights()
        print(str(datetime.now()), 'Initialization successful, training network...')
        network.train(train_images, train_labels)
        print(str(datetime.now()), 'Training ended')
        train_result = network.test(train_images, train_labels)
        print(str(datetime.now()), 'Training data result:', train_result)
        test_result = network.test(test_images, test_labels)
        print(str(datetime.now()), 'Test data precision:', test_result)
if __name__ == "__main__":
    main(sys.argv)
