import numpy as np
import random
import tsplib95
import os
import csv
import time
from parameter_bundles import best_experiment_parameters, basic_experiment_parameters, population_size_experiment_parameters, mutation_rate_experiment_parameters, number_of_generations_experiment_parameters
import multiprocessing

#   ----------------------------------------------------------------------------------------------------------------------------------------
#   PROBLEM SETUP


def read_file(file_name):
    path = "data/datasets/" + file_name + ".tsp"
    return tsplib95.load(path)


def create_distance_matrix(problem):
    nodes = list(problem.get_nodes())
    n = len(nodes)
    distance_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = problem.get_weight(i+1, j+1)
    return distance_matrix


def read_optimal_tour(file_name):
    path = "data/solutions/" + file_name + ".opt.tour"
    with open(path, 'r') as file:
        lines = file.readlines()
        
    tour_start = lines.index("TOUR_SECTION\n") + 1                          # Find the TOUR_SECTION keyword to locate the start of the tour
    tour_end = lines.index("-1\n", tour_start)                              # TOUR_SECTION ends with '-1'
    
    optimal_tour = [int(node) - 1 for node in lines[tour_start:tour_end]]   # Read the tour sequence, converting to 0-based indexing
    
    dimension = len(optimal_tour)                                           # Ensure that all indices are within bounds (0 to DIMENSION - 1)
    if any(node < 0 or node >= dimension for node in optimal_tour):
        raise ValueError("Optimal tour contains invalid node indices.")
    
    return optimal_tour


def calculate_tour_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i + 1]]
    # Add distance to return to the starting city
    total_distance += distance_matrix[tour[-1], tour[0]]
    return total_distance


def percentage_of_optimal(best_distance, optimal_distance):
    if best_distance == 0:
        raise ValueError("Best distance cannot be zero.")

    percentage = (optimal_distance / best_distance) * 100
    return percentage


def initialize_problem_details(file_name):
    if file_name is None:
        raise ValueError("File name not provided.")
    problem = read_file(file_name)
    distance_matrix = create_distance_matrix(problem)
    optimal_tour = read_optimal_tour(file_name)
    optimal_distance = calculate_tour_distance(optimal_tour, distance_matrix)
    best_tour = None
    best_distance = float('inf')
    return problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance

#   ----------------------------------------------------------------------------------------------------------------------------------------
#   DETERMINISTIC ALGORITHMS    


def run_RANDOM_ALG(file_name, output_file_name, TIME_LIMIT):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    counter = 0
    routine_time = 0
    elapsed_time = 0

    start_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time + routine_time >= 1.05 * TIME_LIMIT:
            break

        random_tour = random.sample(range(number_of_cities), number_of_cities)
        random_distance = calculate_tour_distance(random_tour, distance_matrix)
        if random_distance < best_distance:
            best_distance = random_distance
            best_tour = random_tour
        if random_distance == optimal_distance:
            print(f"Optimal distance for random search reached at iteration {counter}, runtime {elapsed_time}")
            break
        print(f"Random algorithm iteration {counter} and runtime {elapsed_time} ")
        counter += 1
        end_of_routine_time = time.time()
        routine_time = end_of_routine_time - current_time

    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour, optimal_distance, percentage_of_optimal(best_distance, optimal_distance),
        f"{time.time() - start_time:.3f}", 'N/A', 'N/A', 'N/A', counter, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        output_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, percentage_of_optimal(best_distance, optimal_distance), 
        f"{time.time() - start_time:.3f}", 'N/A', 'N/A', 'N/A', counter, TIME_LIMIT
    )



def run_GREEDY_ALG(file_name, output_file_name, TIME_LIMIT):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    counter = 0
    routine_time = 0
    elapsed_time = 0

    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time + routine_time >= 1.05 * TIME_LIMIT:
            break

        start_city = random.randint(0, number_of_cities - 1)
        greedy_tour = [start_city]
        unvisited = set(range(number_of_cities)) - {start_city}
        
        # Greedily build the tour
        current_city = start_city
        while unvisited:
            next_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
            greedy_tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        current_distance = calculate_fitness(greedy_tour, distance_matrix)

        if current_distance < best_distance:
            best_distance = current_distance
            best_tour = greedy_tour

        if best_distance == optimal_distance:
            print(f"Optimal distance for greedy search reached at iteration {counter}, runtime {elapsed_time}")
            break
        print(f"Greedy algorithm iteration: {counter} and runtime {elapsed_time}")
        counter += 1
        end_of_routine_time = time.time()
        routine_time = end_of_routine_time - current_time

    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour, optimal_distance, percentage_of_optimal(best_distance, optimal_distance),
        f"{time.time() - start_time:.3f}", 'N/A', 'N/A', 'N/A', counter, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        output_file_name, file_name, number_of_cities,
        optimal_distance, best_distance, percentage_of_optimal(best_distance, optimal_distance),
        f"{time.time() - start_time:.3f}", 'N/A', 'N/A', 'N/A', counter, TIME_LIMIT
    )

#   ----------------------------------------------------------------------------------------------------------------------------------------


#   ----------------------------------------------------------------------------------------------------------------------------------------
#   EVOLUTIONARY ALGORITHM STUFF


#   INITIALIZATION - - - - -

def initialization_random(number_of_cities, POPULATION_SIZE):
    population = []
    for _ in range(POPULATION_SIZE):
        tour = list(range(number_of_cities))
        random.shuffle(tour)
        population.append(tour)
    return population


def initialization_greedy(number_of_cities, distance_matrix, POPULATION_SIZE):
    population = []
    
    for _ in range(POPULATION_SIZE):
        # Start with a random city
        start_city = random.randint(0, number_of_cities - 1)
        tour = [start_city]
        not_visited = set(range(number_of_cities))
        not_visited.remove(start_city)
        
        # Build the tour by choosing the nearest not-visited city at each step
        current_city = start_city
        while not_visited:
            # Find the nearest unvisited city
            nearest_city = min(not_visited, key=lambda city: distance_matrix[current_city][city])
            tour.append(nearest_city)
            not_visited.remove(nearest_city)
            current_city = nearest_city
        
        population.append(tour)
    return population


#   FITNESS - - - - -

def calculate_fitness(tour, distance_matrix):
    fitness = 0
    for i in range(len(tour)-1):
        fitness += distance_matrix[tour[i]][tour[i+1]]
    fitness += distance_matrix[tour[-1]][tour[0]] # return to the starting point of the route
    return fitness


#  SELECTION - - - - -

def selection_tournament(population, distance_matrix, TOURNAMENT_SIZE): 
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: calculate_fitness(x, distance_matrix))  # shouldn't i mark which tours were already used as parents?
    return tournament[0]


# def selection_tournament_ELITISM(population, distance_matrix, TOURNAMENT_SIZE):
#     elite_size = max(1, int(0.05 * len(population)))  # 5% of the population
#     elite = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))[:elite_size]
    
#     tournament = random.sample(population, TOURNAMENT_SIZE)
#     tournament.sort(key=lambda x: calculate_fitness(x, distance_matrix))
    
#     return elite + tournament[:TOURNAMENT_SIZE - elite_size]


# def selection_roulette_wheel(population, distance_matrix):
#     elite_size = max(1, int(0.05 * len(population)))  # 5% of the population
#     elite = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))[:elite_size]
    
#     fitness_values = [1 / calculate_fitness(tour, distance_matrix) for tour in population]
#     total_fitness = sum(fitness_values)
#     selection_probs = [fitness / total_fitness for fitness in fitness_values]
    
#     selected = np.random.choice(population, size=len(population) - elite_size, p=selection_probs, replace=True)
#     return elite + selected.tolist()

# def selection_roulette_wheel_ELITISM(population, distance_matrix):
#     elite_size = max(1, int(0.05 * len(population)))  # 5% of the population
#     elite = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))[:elite_size]
    
#     fitness_values = [1 / calculate_fitness(tour, distance_matrix) for tour in population]
#     total_fitness = sum(fitness_values)
#     selection_probs = [fitness / total_fitness for fitness in fitness_values]

#     # Select indices based on the probabilities, then get the corresponding tours
#     selected_indices = np.random.choice(len(population), size=len(population) - elite_size, p=selection_probs, replace=True)
#     selected = [population[i] for i in selected_indices]
#     return elite + selected

# def selection_roulette_wheel(population, distance_matrix):
#     fitness_values = [1 / calculate_fitness(tour, distance_matrix) for tour in population]
#     total_fitness = sum(fitness_values)
#     selection_probs = [fitness / total_fitness for fitness in fitness_values]

#     # Select index based on the probabilities, then get the corresponding tour
#     selected_index = np.random.choice(len(population), p=selection_probs)
#     return population[selected_index]
def selection_roulette_wheel(population, distance_matrix, num_to_select):
    fitness_values = [1 / calculate_fitness(tour, distance_matrix) for tour in population]
    total_fitness = sum(fitness_values)
    selection_probs = [fitness / total_fitness for fitness in fitness_values]

    # Select multiple parents at once to reduce redundant calculations
    selected_indices = np.random.choice(len(population), size=num_to_select, p=selection_probs, replace=True)
    selected_tours = [population[i] for i in selected_indices]
    
    return selected_tours


#   CROSSOVER - - - - -

def crossover_OX(parent1, parent2): #   ordered crossover
    size = len(parent1)
    child = [-1] * size

    start, end = sorted(random.sample(range(size), 2))  # select random segnemnt of parent1 genotype
    if start > end:
        start, end = end, start
    child[start:end] = parent1[start:end]

    fill_pos = end
    for vertex in parent2:  # fill the rest of the child with the remaining vertices from parent2
        if vertex not in child:
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = vertex
            fill_pos += 1
    return child


# def crossover_PMX(parent1, parent2): #   partially matched crossover
#     size = len(parent1)
#     child = [-1] * size

#     start, end = sorted(random.sample(range(size), 2))  # select random segment of parent1 genotype
#     if start > end:
#         start, end = end, start
#     child[start:end] = parent1[start:end]

#     mapping = {parent1[i]: parent2[i] for i in range(start, end)}

#     for i in range(start, end):
#         if parent2[i] not in child:
#             pos = i
#             while start <= pos < end:
#                 pos = parent1.index(mapping[parent2[pos]])
#             child[pos] = parent2[i]

#     for i in range(size):
#         if child[i] == -1:
#             child[i] = parent2[i]

#     return child


def crossover_CX(parent1, parent2): #   cycle crossover
    size = len(parent1)
    child = [-1] * size  # Initialize child with -1 to indicate unassigned positions
    visited = [False] * size  # Keep track of visited positions

    # Start with the first cycle
    cycle_start = 0
    while cycle_start < size:
        if child[cycle_start] == -1:  # If the position is unassigned
            # Start a new cycle
            current = cycle_start
            while not visited[current]:
                child[current] = parent1[current]  # Assign from parent1
                visited[current] = True
                current = parent2.index(parent1[current])  # Move to the next position in parent2
            
        # Move to the next unvisited position
        cycle_start += 1

    # Fill the remaining positions with genes from parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child
 

#  MUTATION - - - - -

def mutation_swap(tour, MUTATION_RATE): 
    new_tour = tour[:]
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def mutation_inverse(tour, MUTATION_RATE):
    new_tour = tour[:]
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(len(tour)), 2))
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour


#   NEW GENERATION LOOPS - DIFFERENT MODES - - - - -


def create_new_generation_BEST_CASE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        #  PARENT SELECTION
        parent1 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        parent2 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        #  CROSSOVER    
        child = crossover_CX(parent1, parent2)
        #  MUTATION
        child = mutation_inverse(child, MUTATION_RATE)
        
        new_population.append(child)
    return new_population


def create_new_generation_BASE_CASE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        #  PARENT SELECTION
        parent1 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        parent2 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        #  CROSSOVER    
        child = crossover_OX(parent1, parent2)
        #  MUTATION
        child = mutation_swap(child, MUTATION_RATE)
        
        new_population.append(child)
    return new_population


def create_new_generation_SELECTION_ROULETTE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    population_size = len(population)
    new_population = []

    selected_parents = selection_roulette_wheel(population, distance_matrix, population_size * 2)
    
    # Iterate through pairs of parents to create children
    for i in range(0, population_size * 2, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        
        # CROSSOVER
        child = crossover_OX(parent1, parent2)
        # MUTATION
        child = mutation_swap(child, MUTATION_RATE)
        
        new_population.append(child)
    
    return new_population


def create_new_generation_CROSSOVER_CX(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        #  PARENT SELECTION
        parent1 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        parent2 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        #  CROSSOVER    
        child = crossover_CX(parent1, parent2)
        #  MUTATION
        child = mutation_swap(child, MUTATION_RATE)
        
        new_population.append(child)
    return new_population


def create_new_generation_MUTATION_INVERSE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        #  PARENT SELECTION
        parent1 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        parent2 = selection_tournament(population, distance_matrix, TOURNAMENT_SIZE)
        #  CROSSOVER    
        child = crossover_OX(parent1, parent2)
        #  MUTATION
        child = mutation_inverse(child, MUTATION_RATE)
        
        new_population.append(child)
    return new_population

#   ----------------------------------------------------------------------------------------------------------------------------------------
#   MAIN FUNCTIONS - RUN THE ALGORITHM


def run_BEST_CASE(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT, results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_greedy(number_of_cities, distance_matrix,POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()

        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1
        
        population = create_new_generation_BASE_CASE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time

    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


def run_BASE_CASE(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT, results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_random(number_of_cities, POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()

        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1
        
        population = create_new_generation_BASE_CASE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time

    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


def run_SELECTION_ROULETTE(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT,results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_random(number_of_cities, POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()

        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1

        population = create_new_generation_SELECTION_ROULETTE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time

    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


def run_CROSSOVER_CX(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT, results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_random(number_of_cities, POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()
    
        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1

        population = create_new_generation_CROSSOVER_CX(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time

    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


def run_MUTATION_INVERSE(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT, results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_random(number_of_cities, POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()
    
        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1

        population = create_new_generation_MUTATION_INVERSE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time


    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


def run_INITIALIZATION_GREEDY(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT, results_file_name):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialization_greedy(number_of_cities, distance_matrix, POPULATION_SIZE)
    generation_count = 0
    creating_generation_runtime = 0

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time + creating_generation_runtime >= 1.05 * TIME_LIMIT:
            print(f"Time limit reached at Generation {generation}, at runtime {elapsed_time}. Stopping the algorithm.")
            break
        generation_start_time = time.time()

        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        generation_count += 1

        population = create_new_generation_BASE_CASE(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
        generation_end_time = time.time()
        creating_generation_runtime = generation_end_time - generation_start_time

    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )
    save_tsp_results_to_csv(
        results_file_name, file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, generation_count, TIME_LIMIT
    )


#   ----------------------------------------------------------------------------------------------------------------------------------------
#   PRINTING AND SAVING RESULTS


def print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT
        ):
    print(
        f"\n- - - - - - - Evolutionary Algorithm - - - - - - -\n"
        f"Parameters:\n"
        f"   * Population size: {POPULATION_SIZE}\n"
        f"   * Tournament size: {TOURNAMENT_SIZE}\n"
        f"   * Mutation rate: {MUTATION_RATE}\n\n"
        f"Number of cities: {number_of_cities}\n"
        f"Best Tour: {best_tour}\n"
        f"Best Distance: {best_distance}\n"
        f"Optimal Tour: {optimal_tour}\n"
        f"Optimal Distance: {optimal_distance}\n"
        f"Success ratio: {success_ratio:.3f}%\n"
        f"Runtime: {runtime} seconds\n"
        f"Number of generations: {NUMBER_OF_GENERATIONS}\n"
        f"Time limit: {TIME_LIMIT} seconds\n"
    )



def save_tsp_results_to_csv(
        file_name, tsp_instance_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime,
        population_size, tournament_size, mutation_rate, number_of_generations, time_limit
        ):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                'TSP Instance', 'Number of Cities', 'Optimal Distance', 'My Distance', 'Success Rate (%)', 'Runtime (s)',
                'PARAMETERS', 'POPULATION_SIZE', 'TOURNAMENT_SIZE', 'MUTATION_RATE', 'NUMBER_OF_GENERATIONS', 'TIME_LIMIT'
            ])
        writer.writerow([
            tsp_instance_name, number_of_cities, optimal_distance, best_distance, success_ratio, runtime,
            '',  # Empty "PARAMETERS" column
            population_size, tournament_size, mutation_rate, number_of_generations, time_limit
        ])


def sort_csv_file(file_name):
    if not os.path.isfile(file_name):
        print(f"The file '{file_name}' does not exist.")
        return
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader) 
        rows = list(reader)    

    # Sort rows first by 'Number of Cities' (index 1) and then by 'TSP Instance' (index 0)
    rows.sort(key=lambda x: (int(x[1]), x[0]))

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  
        writer.writerows(rows)   


#   ----------------------------------------------------------------------------------------------------------------------------------------


def run_best_experiment():
    names = os.listdir("data/datasets")
    names = [name.replace(".tsp", "") for name in names]
    for name in names:
        for bundle in best_experiment_parameters:
            POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
            run_BEST_CASE(
                name, 
                POPULATION_SIZE,
                TOURNAMENT_SIZE,
                MUTATION_RATE,
                NUMBER_OF_GENERATIONS,
                TIME_LIMIT,
                "results_best.csv"
            )    
    print("\n- - - - Best-case experiment finished! - - - -\n")


#   BASIC EXPERIMENT    
#   Running the EA over 6 different tsp instances, for the same amount of time (6 runs in total):
#       * random init
#       * tournament selection
#       * OX crossover
#       * swap mutation
def run_basic_experiment():
    names = os.listdir("data/datasets")
    names = [name.replace(".tsp", "") for name in names]
    for name in names:
        for bundle in basic_experiment_parameters:
            POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
            run_BASE_CASE(
                name, 
                POPULATION_SIZE,
                TOURNAMENT_SIZE,
                MUTATION_RATE,
                NUMBER_OF_GENERATIONS,
                TIME_LIMIT,
                "results_basic.csv"
            )
    print("\n- - - - Basic experiment finished! - - - -\n")


# EXPERIMENT FOR TESTING THE INFLUENCE OF POPULATION SIZE ON THE ALGORITHM
# Running the EA over one medium-sized tsp instance ('gr202'), over 5 different population sizes (6 runs in total):
#   * random init
#   * tournament selection
#   * OX crossover
#   * swap mutation
def run_population_size_experiment():
    name = 'gr202'
    for bundle in population_size_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_population_size.csv"
        )
    print("\n- - - - Population size experiment finished! - - - -\n")

# EXPERIMENT FOR TESTING THE INFLUENCE OF NUMBER OF GENERATIONS ON THE ALGORITHM
    # Running the EA over one medium-sized tsp instance ('gr202'), over 5 different number of generations (5 runs in total):
    #   * random init
    #   * tournament selection
    #   * OX crossover
    #   * swap mutation
def run_number_of_generations_experiment():
    name = 'gr202'
    for bundle in number_of_generations_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_num_generations.csv"
        )
    print("\n- - - - Number of generations experiment finished! - - - -\n")


#   EXPERIMENT FOR TESTING THE INFLUENCE OF MUTATION RATE ON THE ALGORITHM
#   Running the EA over one medium-sized tsp instance ('gr202'), over 5 different mutation rates (5 runs in total):
#       * random init
#       * tournament selection
#       * OX crossover
#       * swap mutation
def run_mutation_rate_experiment():
    name = 'gr202'
    for bundle in mutation_rate_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_mutation_rate.csv"
        )
    print("\n- - - - Mutation rate experiment finished! - - - -\n")

#   EXPERIMENT FOR TESTING THE INFLUENCE OF 2 DIFFERENT INITIALIZATION STRATEGIES ON THE ALGORITHM (RANDOM AND GREEDY)
#   Running the EA over one medium-sized tsp instance ('gr202') for the same amount of time, with different initialization strategies (2 runs in total):
#       * random init + greedy init
#       * tournament selection
#       * OX crossover
#       * swap mutation
def run_initialization_experiment():
    name = 'gr202'
    for bundle in basic_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_initializations.csv"
        )
        run_INITIALIZATION_GREEDY(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_initializations.csv"
        )
    print("\n- - - - Initialization experiment finished! - - - -\n")


#   EXPERIMENT FOR TESTING THE INFLUENCE OF 2 DIFFERENT SELECTION STRATEGIES ON THE ALGORITHM (TOURNAMENT AND ROULETTE WHEEL)
#   Running the EA over one medium-sized tsp instance ('gr202') for the same amount of time, with different selection strategies (2 runs in total):
#       * random init
#       * tournament selection + roulette wheel selection
#       * OX crossover
#       * swap mutation
def run_selection_experiment():
    name = 'gr202'
    for bundle in basic_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_selections.csv"
        )
        run_SELECTION_ROULETTE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_selections.csv"
        )
    print("\n- - - - Selection experiment finished! - - - -\n")


#   EXPERIMENT FOR TESTING THE INFLUENCE OF 2 DIFFERENT CROSSOVER STRATEGIES ON THE ALGORITHM (OX AND CX)
#   Running the EA over one medium-sized tsp instance ('gr202') for the same amount of time, with different crossover strategies (2 runs in total):
#       * random init
#       * tournament selection
#       * OX crossover + CX crossover
#       * swap mutation
def run_crossover_experiment():
    name = 'gr202'
    for bundle in basic_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_crossovers.csv"
        )
        run_CROSSOVER_CX(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_crossovers.csv"
        )
    print("\n- - - - Crossover experiment finished! - - - -\n")


#   EXPERIMENT FOR TESTING THE INFLUENCE OF 2 DIFFERENT MUTATION STRATEGIES ON THE ALGORITHM (SWAP AND INVERSE)
#   Running the EA over one medium-sized tsp instance ('gr202') for the same amount of time, with different mutation strategies (2 runs in total):
#       * random init
#       * tournament selection
#       * OX crossover
#       * swap mutation + inverse mutation
def run_mutation_experiment():
    name = 'gr202'
    for bundle in basic_experiment_parameters:
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
        run_BASE_CASE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_mutations.csv"
        )
        run_MUTATION_INVERSE(
            name, 
            POPULATION_SIZE,
            TOURNAMENT_SIZE,
            MUTATION_RATE,
            NUMBER_OF_GENERATIONS,
            TIME_LIMIT,
            "results_mutations.csv"
        )
    print("\n- - - - Mutation experiment finished! - - - -\n")

#   EXPERIMENT FOR COMPARING THE PERFORMANCE OF THE EVOLUTIONARY ALGORITHM WITH 2 NON-EVOLUTIONARY ALGORITHMS
#   Running the EA over one medium-sized tsp instance ('gr202') for the same amount of time, with different strategies involved (basic, greedy init, roulette, CX, inverse mutation) (5 runs in total):
# +
#   Running 2 non-evolutionary algorithms over the same tsp instances for the same amount of time (2 runs in total):
def run_non_ea_experiment():
    names = ['berlin52', 'gr202', 'pa561']
    for name in names:
        for bundle in basic_experiment_parameters:
            POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT = bundle.values()
            run_BASE_CASE(name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS, TIME_LIMIT,"results_non_ea.csv")
            run_RANDOM_ALG(name, "results_non_ea.csv", TIME_LIMIT)
            run_GREEDY_ALG(name, "results_non_ea.csv", TIME_LIMIT)
    print("\n- - - - Non-EA experiment finished! - - - -\n")


#   ----------------------------------------------------------------------------------------------------------------------------------------


def main():
    start_time = time.time()

    experiments = [
        run_basic_experiment,
        run_population_size_experiment,
        run_mutation_rate_experiment,
        run_number_of_generations_experiment,
        run_initialization_experiment,
        run_selection_experiment,
        run_crossover_experiment,
        run_mutation_experiment,
        run_non_ea_experiment,
        run_best_experiment
    ]
    processes = [multiprocessing.Process(target=experiment) for experiment in experiments]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    end_time = time.time()
    total_time = end_time - start_time
    seconds = int(total_time % 60)
    minutes = int(total_time // 60)
    print("All experiments have completed.")
    print(f"Total runtime: {minutes} minutes {seconds} seconds")


    sort_csv_file("results_basic.csv")
    sort_csv_file("results_population.csv")
    sort_csv_file("results_generations.csv")
    sort_csv_file("results_mutation_rate.csv")
    sort_csv_file("results_initializations.csv")
    sort_csv_file("results_selections.csv")
    sort_csv_file("results_crossovers.csv")
    sort_csv_file("results_mutations.csv")
    sort_csv_file("results_non_ea.csv")
    sort_csv_file("results_best.csv")


if __name__ == "__main__":
    main()
