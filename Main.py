import streamlit as st
from GAEngine import run,initial_population
import pandas as pd
from ChromosomeMapping import get_gene_mapping,layer_size_search_space,activation_function_search_space,max_iterations_search_space

header = st.container()
body = st.container()

data = pd.read_csv('./data/train.csv')

with header:
    header.title("SCOA mini project")
    header.subheader("Heart Risk Prediction Using NN - Hyperparameter  optimization using GA")
    header.image('./assets/heart.jpg')
    header.subheader("What is Heart Risk Prediction?")
    header.write("We use various parameters like the age,sex,cholesterol and other vitals of a person to predict how likely it is for a person to develop a heart risk. We are using Neural Network for this purpose")
    header.write("[Dataset](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)")
    
    header.subheader("What is Genetic Algorithm?")
    header.write("Genetic Algorithm is a search algorithm which is used to solve NP hard problem. It is based on the natural process of evolution and believes in the term 'Survival of the fittest'.")
    header.write("In a GA we have a set of initial agents, who will cross breed to form next generation of agents with certain mutations. Each time only the fittest agents will be selected for cross breeding so as to move in a forward direction to optimize the cost to find the solution.")
    
    header.subheader("How will GA help in finding optimal hyperparameter configuration for the Neural Network?")
    header.write("Consider GA as a sandbox for generating various Neural Networks. Each neural network will have a chromose which tell the GA engine to generate a NN with a particular hyper parameter configuration. Then we test the performance of the neural networks using the objective function (accuracy). Fittest NN i.e. NN with highest accuracy will be selected to generate the next generation of the NN. Over mutliple generations we will get an optimized configuration of the hyperparameters.")
    header.image('./assets/GA_Engine.png')
    
    header.subheader("Chromosome Mapping and Mutation probabilities")
    header.write("Each Neural Network is defined as a function of neuron_layers,activation_function and max_itr. Each chromosome is a real valued vector representing a single value from a predefined set of layers, activation functions and max_iterations")
    header.write("NN shapes")
    header.write(layer_size_search_space)
    header.write("Activation Functions")
    header.write(activation_function_search_space)
    header.write("max iterations")
    header.write(max_iterations_search_space)
    
    header.subheader("Dataset description:")
    desc = data.describe()
    header.write(desc)
        
    header.subheader("Sample Data:")
    header.write(data.head())
    
    header.subheader("Initial Population")
    header.write("Initial population is the set of chromosomes that will start our evolution process. These chromosome are vectors of type [neural_network_shape,max_itr,activation_function]")
    population=[]
    for i in initial_population:
        population.append(get_gene_mapping(i))
    header.write(population)

with body:
    body.title("Search Optimal configuration")
    if body.button("Run"):
        body.write("The GA Engine is running! you can check the training logs on the Console.")
        results = run()
        file_path = results[4] 
        body.write("All configuration of the trained NN is present in the file: "+file_path)   
        logs = pd.read_csv("./"+file_path)
        body.write(logs)
        
        body.subheader("Final Results")
        body.write("Done!!! Check out the optimal configuration for you neural network")
        body.write("The System evolved for "+str(results[3])+" generations")
        body.write("Global Best Accuracy:"+str(results[0]))
        body.write("Neural Network:"+str(results[1]))
        body.write("Neural Network Configuration:"+str(results[2]))
    pass