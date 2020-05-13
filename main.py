import logging
from genetic_algorithm import GA
from tqdm import tqdm
import csv
import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log.txt'
)

def train_networks(networks, dataset):
    
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset):
  
    evolver = GA(nn_param_choices)
    networks = evolver.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = evolver.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: (x.accuracy,-x.loss), reverse=True)
    # Print out the top 5 networks.
    print_networks(networks[:5])



def print_networks(networks):
    """
        Print top 5 networks
    """
    logging.info('-'*80)
    logging.info('Top 5 networks')
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate_top_network_file(networks):
    with open("meta/top_10.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(networks)


def main(dataset='data-original'):
    """Evolve a network."""
    generations = 12  
    population = 20  # networks
    # dataset = 'data-small'
    # dataset = 'data-original'

    nn_param_choices = {
        'nb_layers': [2, 3, 4, 5, 6],
        'nb_dense_layers' : [1, 2, 3],
        'filters' : [16, 24, 32, 48, 64],
        'kernel_sizes' : [2,3,4,5,6,7],
        'nb_neurons': [128,192, 256,384, 512],
        'activation': ['relu', 'elu', 'selu', 'sigmoid'],
        'optimizer': ['sgd', 'adam', 'adamax', 'nadam', 'adagrad', 'adadelta'],
        'dropout' : [0.1, 0.2, 0.3, 0.4, 0.5 ]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main(sys.argv[1])
