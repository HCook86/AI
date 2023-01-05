#See explanation.txt for code explanations

# SOURCE 
# Math and explanations -->     https://www.3blue1brown.com/topics/neural-networks
# Numpy documentation -->     https://numpy.org/doc/stable/
# Progress bar  --> https://github.com/rsalmei/alive-progress

import numpy as np 
import logging
from utils import NumpyEncoder
from alive_progress import alive_bar

#Set up basic logging module configuration
logging.basicConfig(filename='history.log', encoding='utf-8', level=logging.DEBUG)

class Network():

    # Generate random weights and biases in specific format
    def generate_data(self, layers_nodes):
        data = list()
        layer_number = 0
        input_layer = layers_nodes.pop(0)

        for layer in layers_nodes:
            layer_data = tuple()
            if layer_number == 0:
                layer_data = (np.random.random((layer, input_layer)), np.random.random((layer, 1)))
                
            else:
                layer_data = (np.random.random((layer, layers_nodes[layer_number-1])), np.random.random((layer, 1)))
            data.append(layer_data)
            layer_number += 1
        logging.debug("RANDOM WEIGHTS AND BIASES GENERATED: \n ")
        return data


    # Initialization function. Generate all the data randomly on object creation and store shape in object
    def __init__(self, node_structure):
        self.input_layer = node_structure[0]
        self.structure = node_structure
        self.data = self.generate_data(node_structure)


    # Save current data to a .txt file
    def save_to_file(self, file_name):
        logging.debug("DATA SAVED TO " + file_name + "\n-----------------------\n")
        np.savetxt(file_name, self.data_to_vector())


    # Load data from specified .txt file
    def load_from_file(self, file_name):
        logging.debug("DATA LOADED FROM " + file_name + "\n------------------------\n")
        self.data = self.vector_to_data(np.loadtxt(file_name))


    # Define the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    # Flatten all data into a vector to do gradient descent.
    def data_to_vector(self):
        input = self.data
        output = np.empty(0)

        # Append all weights
        for element in input:
            output = np.append(output, element[0].flatten())
        # Append all biases
        for element in input:
            output = np.append(output, element[1])
        return output


    # Turn 1 dimensional vector into format used by run function 
    def vector_to_data(self, vector):
        weights = list()
        layer_number = 0
        # Weights
        for element in self.structure:
            if layer_number == 0:
                # Get first layer weights
                weights.append(vector[:(self.input_layer*element)].reshape(element,self.input_layer))
                # Remove them from original vector
                vector = vector[(self.input_layer*element):]

            else:
                weights.append(vector[:(self.structure[layer_number-1]*element)].reshape(element,self.structure[layer_number-1]))
                vector = vector[(self.structure[layer_number-1]*element):]
            layer_number += 1
        
        biases = list()
        # Biases
        for element in self.structure:
            biases.append((vector[:element].reshape(element,1)))
            vector = vector[(element):]

        data = list()
        # Arrange all weights and biases into normal dataa format. [(weights, biases), (weights, biases), (weights, biases)]
        for layer in range(0, len(weights)):
            data.append((weights[layer], biases[layer]))
        return data


    # Check the input format of the data (more information in explanation.txt)
    def check(self, weights, values, biases):
        if weights.shape[0] == biases.shape[0]: 
            if weights.shape[1] == values.shape[0]:
                logging.debug("CHECK OK")
                logging.debug("\n\n---------------------------\n")
                return True
            else:
                logging.critical("CHECK FAILED: (weights and values don't match)")
                logging.debug("\n\n---------------------------\n")
                print(str(weights))
                print(str(biases))
                raise ValueError("Format Check Failed")
        else: 
            logging.critical("CHECK FAILED: (weights and biases don't match)")
            logging.debug("\n\n---------------------------\n")
            print(str(weights))
            print(str(biases))
            raise ValueError("Format Check Failed")


    # Run individual layer
    def layer(self, weights, values, biases):
        # Log process
        # logging.debug("LAYER INFO: \n    Input-Nodes: " + str(weights.shape[1]) + "\n    Output-Nodes:" + str(biases.shape[0]) +"\n    weights: \n" + str(weights) + "\n    values: \n" + str(values) +  "\n    biases: \n" + str(biases))
        
        # Multiply the matrices and add the biases
        calc = np.add((weights @ values), biases)

        # Apply to sigmoid function to all elementes of matrix
        sigfunc = np.vectorize(self.sigmoid)
        ans = sigfunc(calc)
        
        logging.debug("ANSWER: \n" + str(ans))
        logging.debug("\n\n---------------------------\n")
        return ans


    # Calculate cost of individual run. More information in explanation.txt
    def error_for_run(self, expected, output):
        error = expected - output
        error = np.sum(np.square(error))
        logging.debug("COST CALCULATION: \n    Expected:    " + str(expected) + "\n    Output:    " + str(output) + "\n    Error:     " + str(error) + "\n\n---------------------------\n")
        return error


    # Main function to run all layers of the network. Only takes input values as argument, and gets others from self.data
    # Data has the following structure:     layer 1             layer2              layer3
    #                                   [(weights, biases),(weights, biases),(weights, biases)]
    def run(self, initial_values):
        data = self.data
        logging.debug("\n\n ------------------------------------ RUN " + "-------------------------------------------\n")

        # current_values variable is given the first time as user input. Then the values are stored from the previous layer
        current_values = initial_values
        layer_number = 0
        layer_outputs = list()

        # Iterate through each layer of the network
        for layer_data in data:
            layer_number += 1
            # Check and run each layer
            logging.debug("RUNNING FORMAT CHECK ON LAYER: " + str(layer_number))
            self.check(layer_data[0], current_values, layer_data[1])

            logging.debug("RUNNING LAYER: " + str(layer_number))
            current_values = self.layer(layer_data[0],  current_values, layer_data[1])
            
            # Store the output of each layer and print the last one (output layer)
            layer_outputs.append(current_values)        
        return layer_outputs[-1]





    # Function to train the AI
    def train(self, training_data):
        with alive_bar(len(training_data)) as bar:
            for element in training_data:
                bar()
        return None

    # Cost function
    def cost(self, data_vector):
        global_cost = None
        return global_cost


net = Network([784,16,16,10])

print(net.data)

#net.train([1])

#net.run(net.generate(784,1))

#net.cost(net.run(np.random.random((784, 1))), np.array([0,0,0,0,0,0,0,0,0,1]))

#net.cost(net.run([(net.generate(16,784), net.generate(16,1)), ((net.generate(16,16), net.generate(16,1))), (net.generate(10,16), net.generate(10,1))], net.generate(784,1)), np.array([0,0,0,0,0,0,0,0,0,1]))


#This line works (1 Dimension) WRONG: MATRIX TRANPOSED
#print(net.layer(np.array([[0, 1, 0]]), np.array([[1], [0], [0]]), np.array([[1]])))
#print(net.check(np.array([[0, 1, 0]]), np.array([[1], [0], [0]]), np.array([[1]])))


#This works (2 Dimensions) CORRECT! MATRICES NOT TRANSPOSED
#net.check(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 0, 0]), np.array([1, 1]))
#net.layer(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 0, 0]), np.array([1, 1]))


#This works (More than one layer)
#print(net.cost(np.array([1]) , net.run([(np.array([[1, 1, 0],[0, 1, 0]]), np.array([[0], [0]])),     (np.array([[1, 1]]), np.array([[1]]))], np.array([[1], [0], [0]]))))


#net.run([(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 1])),     (np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 1]))], np.array([1, 0, 0]))
