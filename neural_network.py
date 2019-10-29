import sys
from datetime import datetime
import numpy as np
from mnist import MNIST

def mix(x, y):
    random_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(random_state)
    np.random.shuffle(y)
    return x, y

def relu(arg):
    return 1 / (1 + np.exp(-arg))

def softmax(arg):
    return np.exp(arg) / np.sum(np.exp(arg), axis=0)

def deriv(func, arg):
    return np.vectorize(relu_der_s)(arg)

def relu_der_s(x):
    return 1 if x > 0 else 0

class NeuralNetwork:
    def __init__(self, hidden_nodes=40, output_nodes=10, learn_rate= 0.1):
        self.input_nodes = 0
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.w1 = np.array([])
        self.w2 = np.array([])
        self.learn_rate = learn_rate
        self.batch_size = 0

    def __initialize_weights(self):
        self.w1 = 2*np.random.rand(self.hidden_nodes, self.input_nodes) -1
        self.w2 = 2*np.random.rand(self.output_nodes, self.hidden_nodes) -1

    def __calc_hidden(self, input):
        self.w1_dot = np.dot(self.w1, input.transpose())
        self.w1_updated = relu(self.w1_dot)

    def __calc_output(self, input):
        self.__calc_hidden(input)
        self.w2_dot = np.dot(self.w2, self.w1_updated)
        self.w2_updated = softmax(self.w2_dot)

    def __back(self, temp_output_layer, output_layer_expected):
        delta2 = output_layer_expected.transpose() - self.w2_updated
        dws  = np.dot(delta2, self.w1_updated.transpose())/self.batch_size

        delta1 = np.dot(self.w2.transpose(), delta2) * deriv(relu, self.w1_dot)
        dwh  = np.dot(delta1, temp_output_layer)/self.batch_size

        self.w2 = self.w2 + self.learn_rate * dws
        self.w1 = self.w1 + self.learn_rate * dwh

    def train(self, data, labels, batch_size, epochs):
        self.batch_size = batch_size
        self.input_nodes = data.shape[1]
        self.__initialize_weights()

        for epoch in range(epochs):
            data, labels = mix(data, labels)
            for i in range(0, data.shape[0], self.batch_size):
                self.__calc_output(data[i:i + self.batch_size])
                self.__back(data[i:i + self.batch_size], labels[i:i + self.batch_size])

    def test(self, data, labels):
        self.__calc_output(data)
        crossentropy = -np.sum(labels * np.log(self.w2_updated.transpose())) / data.shape[0]

        result_net = np.argmax(self.w2_updated, axis=0)
        result_real = np.argmax(labels, axis=1)
        accuracy = (result_net == result_real).mean()

        return crossentropy, accuracy

def read_mnist_data(data_folder, output_nodes):
    mndata = MNIST(data_folder)
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    np_train_labels = np.zeros((len(train_labels), output_nodes), dtype='float32')
    for i in range(len(train_labels)):
        np_train_labels[i][train_labels[i]] = 1
    np_test_labels = np.zeros((len(test_labels), output_nodes), dtype='float32')
    for i in range(len(test_labels)):
        np_test_labels[i][test_labels[i]] = 1
    return np.array(train_images)/255, np_train_labels, np.array(test_images)/255, np_test_labels

def main(argv):
    if len(argv) != 7:
        print("""Usage:
python neural_network.py [data folder] [epochs] [learn rate] [hidden size] [output size] [batch_size]""")
        sys.exit()
    else:
        data_folder = argv[1]
        epochs = int(argv[2])
        learn_rate = float(argv[3])
        hidden_nodes = int(argv[4])
        output_nodes = int(argv[5])
        batch_size = int(argv[6])
        print('Loading data from ', data_folder)
        train_images, train_labels, test_images, test_labels = read_mnist_data(data_folder, output_nodes)
        print('Found', len(train_images), 'training images')
        print('Found', len(test_images), 'testing images')
        network = NeuralNetwork(hidden_nodes, output_nodes, learn_rate)
        network.train(train_images, train_labels, batch_size, epochs)
        print(str(datetime.now()), 'Training ended')
        train_result = network.test(train_images, train_labels)
        print(str(datetime.now()), 'Training data result:', train_result)
        test_result = network.test(test_images, test_labels)
        print(str(datetime.now()), 'Test data precision:', test_result)
if __name__ == "__main__":
    main(sys.argv)
