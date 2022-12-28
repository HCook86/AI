#See explanation.txt for code explanations

import numpy as np 
import logging
import json

#Set up basic logging module configuration
logging.basicConfig(filename='history.log', encoding='utf-8', level=logging.DEBUG)

class Network():

    # Define the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Function should generate random values for a specific format 
    def generate(self, input, output):
        random_data = np.random.random((input, output))
        #logging.debug("RANDOM DATA GENERATED: \n " + str(random_data) +" \n\n---------------------------\n")
        return random_data

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
                raise ValueError("Check Failed")
        else: 
            logging.critical("CHECK FAILED: (weights and biases don't match)")
            logging.debug("\n\n---------------------------\n")
            print(str(weights))
            print(str(biases))
            raise ValueError("Check Failed")


    # Run individual layer
    def layer(self, weights, values, biases):
        # Log process
        logging.debug("LAYER INFO: \n    Input-Nodes: " + str(weights.shape[1]) + "\n    Output-Nodes:" + str(biases.shape[0]) +"\n    weights: \n" + str(weights) + "\n    values: \n" + str(values) +  "\n    biases: \n" + str(biases))
        
        # Multiply the matrices and add the biases
        calc = np.add((weights @ values), biases)

        # Apply to sigmoid function to all elementes of matrix
        sigfunc = np.vectorize(self.sigmoid)
        ans = sigfunc(calc)
        
        logging.debug("ANSWER: \n" + str(ans))
        logging.debug("\n\n---------------------------\n")
        return ans

    def cost(self, expected, output):
        error = expected - output
        error = np.sum(np.square(error))
        logging.debug("COST CALCULATION: \n    Expected:    " + str(expected) + "\n    Output:    " + str(output) + "\n    Error:     " + str(error) + "\n\n---------------------------\n")
        return error

    # Main function to run all layers of the network
    def run(self, data, initial_values):
        logging.debug("\n\n ------------------------------------ ITERATION " + "" + "-------------------------------------------\n")
        # current_values variable is given the first time as user input. Then the values are stored from the previous layer
        current_values = initial_values
        layer_number = 0
        layer_outputs = list()

        # Iterate through each layer of the network
        for layer_data in data:
            layer_number += 1
            # Check and run each layer
            logging.debug("RUNNING CHECK ON LAYER: " + str(layer_number))
            self.check(layer_data[0], current_values, layer_data[1])
            logging.debug("RUNNING LAYER: " + str(layer_number))
            current_values = self.layer(layer_data[0],  current_values, layer_data[1])
            layer_outputs.append(current_values)        
        return layer_outputs[-1]





net = Network()

net.cost(net.run([(net.generate(16,784), net.generate(16,1)), ((net.generate(16,16), net.generate(16,1))), (net.generate(10,16), net.generate(10,1))], net.generate(784,1)), np.array([0,0,0,0,0,0,0,0,0,1]))


#This line works (1 Dimension) WRONG: MATRIX TRANPOSED
#print(net.layer(np.array([[0, 1, 0]]), np.array([[1], [0], [0]]), np.array([[1]])))
#print(net.check(np.array([[0, 1, 0]]), np.array([[1], [0], [0]]), np.array([[1]])))


#This works (2 Dimensions) CORRECT! MATRICES NOT TRANSPOSED
#net.check(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 0, 0]), np.array([1, 1]))
#net.layer(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 0, 0]), np.array([1, 1]))


#This works (More than one layer)
#print(net.cost(np.array([1]) , net.run([(np.array([[1, 1, 0],[0, 1, 0]]), np.array([[0], [0]])),     (np.array([[1, 1]]), np.array([[1]]))], np.array([[1], [0], [0]]))))


#net.run([(np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 1])),     (np.array([[0, 1, 0],[0, 1, 0]]), np.array([1, 1]))], np.array([1, 0, 0]))