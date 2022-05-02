""" 
chromosome: [0,0,0,0]

gene mapping with index:
    0. Layer size
    1. activation function
    2. max iterations
"""

import random

# Constants for gene mapping with model properties
layer_size_search_space = [
    [10],
    [10,20,10],
    [20,25,30],
    [50],
    [50,100],
    [100],
    [15,30,45],
    [32],
    [32, 64, 32],
    [32, 64, 128, 64, 32],
    [64],
    [64, 64],
    [64, 128, 64],
    [64, 128, 128, 64],
    [64, 128, 256, 128, 64]
]
layer_size_flip_probability = 0.5

activation_function_search_space = ['relu', 'tanh', 'logistic']
activation_function_flip_probability = 0.5

max_iterations_search_space = [100, 110, 115, 120, 125, 150, 200, 250, 300, 350]
max_iterations_flip_probability = 0.1


def get_gene_mapping(chromosome):
    # chromosome: array of intergers describing index
    # eg: [2,1,0]
    # 2: layer_size_search_space[2] => [64,64]
    # 1: max_iteration_search_space[1] => 100
    # 0: activation_function_search_space[0] => relu
    
    layers = layer_size_search_space[chromosome[0]]
    max_iterations = max_iterations_search_space[chromosome[1]]
    activation_function = activation_function_search_space[chromosome[2]]

    return layers, max_iterations, activation_function


def mutation(chromosome):

    mutated_chromosome = chromosome

    # layer size mutation
    p = random.random()
    if p < layer_size_flip_probability:
        layer_size_index = random.randrange(0, len(layer_size_search_space)-1)
        mutated_chromosome[0] = layer_size_index

    # max iteration mutation
    p = random.random()
    if p < max_iterations_flip_probability:
        max_itr_index = random.randrange(0, len(max_iterations_search_space)-1)
        mutated_chromosome[1] = max_itr_index

    # activation function mutation
    p = random.random()
    if p < activation_function_flip_probability:
        max_itr_index = random.randrange(0, len(activation_function_search_space)-1)
        mutated_chromosome[2] = max_itr_index

    return mutated_chromosome
