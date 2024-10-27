import numpy as np
import random
import tsplib95
import os
import csv
import time


#   TODO:
#   - mutation: Implement INVERSE mutation
#   - selection: Implement roulette wheel 
#   - crossover: Implement partially matched crossover
#   - solutions: Implement 2 non-evolutionary algorithms


parameter_bundles = [
    {
        "POPULATION_SIZE": 500,
        "TOURNAMENT_SIZE": 2,
        "MUTATION_RATE": 0.05,
        "NUMBER_OF_GENERATIONS": 1000
    },
    {
        "POPULATION_SIZE": 1000,
        "TOURNAMENT_SIZE": 2,
        "MUTATION_RATE": 0.05,
        "NUMBER_OF_GENERATIONS": 1000
    },
    {
        "POPULATION_SIZE": 2000,
        "TOURNAMENT_SIZE": 2,
        "MUTATION_RATE": 0.05,
        "NUMBER_OF_GENERATIONS": 1000
    }, 
    # {
    #     "POPULATION_SIZE": 1000,
    #     "TOURNAMENT_SIZE": 2,
    #     "MUTATION_RATE": 0.01,
    #     "NUMBER_OF_GENERATIONS": 1000
    # },
    # {
    #     "POPULATION_SIZE": 1000,
    #     "TOURNAMENT_SIZE": 2,
    #     "MUTATION_RATE": 0.1,
    #     "NUMBER_OF_GENERATIONS": 1000
    # },
    {
        "POPULATION_SIZE": 1000,
        "TOURNAMENT_SIZE": 2,
        "MUTATION_RATE": 0.05,
        "NUMBER_OF_GENERATIONS": 500
    },
    {
        "POPULATION_SIZE": 1000,
        "TOURNAMENT_SIZE": 2,
        "MUTATION_RATE": 0.05,
        "NUMBER_OF_GENERATIONS": 2000
    },
]

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
#   EVOLUTIONARY ALGORITHM STUFF


def initialize_population(number_of_cities, POPULATION_SIZE):
    population = []
    for _ in range(POPULATION_SIZE):
        tour = list(range(number_of_cities))
        random.shuffle(tour)
        population.append(tour)
    return population


def calculate_fitness(tour, distance_matrix):
    fitness = 0
    for i in range(len(tour)-1):
        fitness += distance_matrix[tour[i]][tour[i+1]]
    fitness += distance_matrix[tour[-1]][tour[0]] # return to the starting point of the route
    return fitness


def tournament_selection(population, distance_matrix, TOURNAMENT_SIZE): #   TODO: implement roulette wheel selection
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: calculate_fitness(x, distance_matrix))  # shouldn't i mark which tours were already used as parents?
    return tournament[0]


def crossover(parent1, parent2):    # TODO: ordered crossover: add cycle crossover and partially matched crossover
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
    

def mutation(tour, MUTATION_RATE):  # TODO: I have swap - implement INVERSE mutation
    new_tour = tour[:]
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def create_new_generation(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE):
    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        parent1 = tournament_selection(population, distance_matrix, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, distance_matrix, TOURNAMENT_SIZE)
        
        child = crossover(parent1, parent2)
        child = mutation(child, MUTATION_RATE)
        new_population.append(child)
    return new_population

#   ----------------------------------------------------------------------------------------------------------------------------------------
#   MAIN FUNCTIONS - RUN THE ALGORITHM


def run(file_name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS):
    problem, distance_matrix, optimal_tour, optimal_distance, best_tour, best_distance = initialize_problem_details(file_name)
    number_of_cities = distance_matrix.shape[0]
    population = initialize_population(number_of_cities, POPULATION_SIZE)

    start_time = time.time()

    for generation in range(NUMBER_OF_GENERATIONS):
        population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
        current_best_distance = calculate_fitness(population[0], distance_matrix)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = population[0]
        
        if best_distance == optimal_distance:
            print(f"Optimal distance reached at Generation {generation}")
            break

        print(f"Generation {generation} - Best Distance: {best_distance}")
        population = create_new_generation(population, distance_matrix, TOURNAMENT_SIZE, MUTATION_RATE)
    
    end_time = time.time()
    runtime = end_time - start_time
    runtime_f = f"{runtime:.3f}"

    success_ratio = percentage_of_optimal(best_distance, optimal_distance)
    print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime_f, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS
    )
    save_tsp_results_to_csv(
        "results.csv", file_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime_f,
        POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS
    )


def print_results(
        number_of_cities, best_tour, best_distance, optimal_tour,
        optimal_distance, success_ratio, runtime, POPULATION_SIZE,
        TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS
        ):
    print("\n- - - - - - - Evolutionary Algorithm - - - - - - -\n")
    print(f'Parameters: \n'
        f'   * Population size: {POPULATION_SIZE}\n'
        f'   * Tournament size: {TOURNAMENT_SIZE}\n'
        f'   * Mutation rate: {MUTATION_RATE}\n'
        f'   * Number of generations: {NUMBER_OF_GENERATIONS}\n')
    print(f"Number of cities: {number_of_cities}")
    print(f"Best Tour: {best_tour}")
    print(f"Best Distance: {best_distance}")
    print(f"Optimal Tour: {optimal_tour}")
    print(f"Optimal Distance: {optimal_distance}")
    print("Success ratio: {:.3f}%".format(success_ratio))
    print(f"Runtime: {runtime} seconds")


def save_tsp_results_to_csv(
        file_name, tsp_instance_name, number_of_cities, 
        optimal_distance, best_distance, success_ratio, runtime,
        population_size, tournament_size, mutation_rate, number_of_generations
        ):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                'TSP Instance', 'Number of Cities', 'Optimal Distance', 'My Distance', 'Success Rate (%)', 'Runtime (s)',
                'PARAMETERS', 'POPULATION_SIZE', 'TOURNAMENT_SIZE', 'MUTATION_RATE', 'NUMBER_OF_GENERATIONS'
            ])
        writer.writerow([
            tsp_instance_name, number_of_cities, optimal_distance, best_distance, success_ratio, runtime,
            '',  # Empty "PARAMETERS" column
            population_size, tournament_size, mutation_rate, number_of_generations
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


def main():
    names = os.listdir("data/datasets")
    names = [name.replace(".tsp", "") for name in names]

    for name in names:
        name = 'bayg29'
        for bundle in parameter_bundles:
            POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS = bundle.values()
            run(name, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_RATE, NUMBER_OF_GENERATIONS)
            break
        break
    sort_csv_file("results.csv")


if __name__ == "__main__":
    main()
