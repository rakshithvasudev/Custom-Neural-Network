import numpy as np

if __name__ == "__main__":
    # initialize a single neuron neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights: ')
    print(neural_network.synaptic_weights)

    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).transpose()

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New Synaptic Weights after training")