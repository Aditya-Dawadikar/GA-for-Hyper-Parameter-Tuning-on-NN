import uuid
import random
import Model as m
from ChromosomeMapping import get_gene_mapping, mutation
import math
from Export import Export   
from operator import itemgetter

data = m.DataSource()
neural_network_list = {}
export_data = []
exporter = Export()

stopping_patience = 5 # stop after 5 consective non evolving generations
select_best_count = 5
initial_population = [
                        [2, 1, 1], 
                        [3, 4, 2], 
                        [4, 2, 0], 
                        [0, 3, 0],
                        [7, 2, 2],
                        [5, 4, 1]
                    ]

def generate_neural_newtwork(chromosome):
    nn_name = uuid.uuid4()
    print("training NN: "+str(nn_name))
    
    layers, max_iterations,activationo_function = get_gene_mapping(chromosome=chromosome)
    nn = m.NeuralNetwork(
                            layers, 
                            max_iterations,
                            activationo_function
                        )
    
    X,y = data.get_train_data()
    nn.train(X,y)
    neural_network_list[nn_name] = nn

    return nn_name

def selectBest(candidates, count):
    if len(candidates)<count:
        return candidates
    else:
        sorted_candidates = sorted(candidates,key=itemgetter(4))
        return sorted_candidates[0:count]
    
    # random candicate selection
    # rand_index = random.randint(0, len(candidates)-1)
    # max = fitness_score(candidates[rand_index])
    # selected = []
    # selected.append(candidates[rand_index])

    # selecting best
    # for i, candidate in enumerate(candidates):
    #     if len(selected) < count and i != rand_index:
    #         score = fitness_score(candidate)
    #         if score > max:
    #             max = score
    #             selected.append(candidate)

    # if len(selected) < count:
    #     new_rand = random.randint(0, len(candidates)-1)
    #     while new_rand == rand_index:
    #         new_rand = random.randint(0, len(candidates)-1)

    #     selected.append(candidates[new_rand])

    # return selected

def cross_over(parent_1, parent_2):
    
    p1,p2=parent_1[0:3],parent_2[0:3]
    
    crossover_point = math.floor(len(p1)/2)
    
    child1 = p1[:crossover_point]
    child1.extend(p2[crossover_point:])

    child2 = p2[:crossover_point]
    child2.extend(p1[crossover_point:])

    return child1, child2

def fitness_score(chromosome):
    if len(chromosome)<5:
        nn = neural_network_list[chromosome[3]]
        X,y = data.get_test_data()
        return nn.evaluate_model(X,y)
    else:
        return chromosome[4]

def getMaxScore(generation,candidates):
    max = 0
    candidate = []

    for i in range(len(candidates)):
        score = fitness_score(candidates[i])
        
        # data for export
        save_model_config(generation, candidates[i],score)
        
        if score > max:
            max = score
            candidate = candidates[i]
    return max, candidate

def save_model_config(generation,chromosome,acc):
    layers, max_iterations,activation_function = get_gene_mapping(chromosome=chromosome)
    export_data.append([generation,chromosome[3], layers, max_iterations, activation_function,acc])

def evolution():

    # initial population
    print("training initial population models")
    population = initial_population
    for i in range(len(population)):
        nn_name = generate_neural_newtwork(population[i])
        population[i].append(nn_name)   # 3rd index hold NN name
        # print(len(population[i]))
        population[i].append(fitness_score(population[i])) # 4th index holds fitness score

    # initial variables
    n_gen = 1
    curr_best = 0
    winner = []
    global_winner = []
    global_best = 0
    p=5

    while True:
        
        if p<=0:
            print("\nProgram terminated because global maxima didnt improve for "+str(stopping_patience)+" consecutive generations")
            break
        
        print("\n------------------ Generation: "+str(n_gen)+" ------------------")
        fittest_parents = selectBest(candidates=population, count=select_best_count)

        new_children = []

        for i in range(len(fittest_parents)):
            for j in range(len(fittest_parents)):
                if i != j:
                    child_1, child_2 = cross_over(
                        parent_1=fittest_parents[i][:len(fittest_parents[i])-1],
                        parent_2=fittest_parents[j][:len(fittest_parents[j])-1],
                    )
                    new_children.append(child_1)
                    new_children.append(child_2)

        mutated_children = []
        for i in range(len(new_children)):
            mutated_chromosome = mutation(new_children[i])
            nn_name = generate_neural_newtwork(mutated_chromosome)
            mutated_chromosome.append(nn_name)
            mutated_chromosome.append(fitness_score(mutated_chromosome))
            
            mutated_children.append(mutated_chromosome)

        population = mutated_children

        curr_best, winner = getMaxScore(n_gen,population)
        if curr_best>global_best:
            global_best = curr_best
            global_winner = winner
            p = stopping_patience
        else :
            p-=1
        
        print("\n Generation: "+str(n_gen))
        print("Local Best: "+str(curr_best))
        print("Suboptimal Configuration: "+str(winner))
        print("Global Best: "+str(global_best))
        n_gen += 1

    print("Best score:"+str(global_best))
    print("Winner:"+str(global_winner))
    print("Optimal configuration:"+str([get_gene_mapping(winner)]))


evolution()
exporter.createDataFrame(export_data)
