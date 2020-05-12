"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        self.accuracy = 0.
        self.loss = 0
        # self.validation_accuracy = 0
        # self.validation_loss=0
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
         # 1 is accuracy. 0 is loss. 4 is validation accuracy. 3 is validation loss
        if self.accuracy == 0.:
            self.loss, self.accuracy  = train_and_score(self.network, dataset)
            

    def print_network(self):
        """Print out a network."""
        logging.info("************************************************")
        logging.info("Network details: ")
        logging.info(self.network)
        logging.info("Accuracy: %.2f%%" % (self.accuracy * 100))
        logging.info("Validation loss: %.2f%%" % (self.loss * 100))
        # logging.info("Validation accuracy: %.2f%%" % (self.validation_accuracy * 100))
        # logging.info("Training loss: %.2f%%" % (self.loss * 100))
