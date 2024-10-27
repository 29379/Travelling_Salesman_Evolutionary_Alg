import numpy as np
import random
import tsplib95

# TODO: find the optimal solution at the beginning, and use it as an alternative exit case different from the number of generations
# POPULATION_SIZE = 500
# TOURNAMENT_SIZE = 2
# MUTATION_RATE = 0.01
# NUMBER_OF_GENERATIONS = 2000

class TSP:
    def __init__(self, population_size, tournament_size, mutation_rate, number_of_generations, problem_name=None):
        self.POPULATION_SIZE = population_size
        self.TOURNAMENT_SIZE = tournament_size
        self.MUTATION_RATE = mutation_rate
        self.NUMBER_OF_GENERATIONS = number_of_generations
        if problem_name is not None:
            self.PROBLEM_PATH = "data/datasets/" + problem_name + ".tsp"
            self.PROBLEM_SOLUTION_PATH = "data/solutions/" + problem_name + ".opt.tour"
        else:
            self.PROBLEM_PATH = None
            self.PROBLEM_SOLUTION_PATH = None


    def read_file(self, file_name):
        path = "data/datasets/" + file_name
        return tsplib95.load(path)
    

    def create_distance_matrix(self, problem):
        nodes = list(problem.get_nodes())
        n = len(nodes)
        distance_matrix = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = problem.get_weight(i+1, j+1)
        # print(distance_matrix)
        return distance_matrix

    
    def read_optimal_tour(filename):
        path = "data/solutions/" + filename
        with open(path, 'r') as file:
            lines = file.readlines()
            
        # Find the TOUR_SECTION keyword to locate the start of the tour
        tour_start = lines.index("TOUR_SECTION\n") + 1
        tour_end = lines.index("-1\n", tour_start)  # TOUR_SECTION ends with '-1'
        
        # Read the tour sequence, converting to 0-based indexing
        optimal_tour = [int(node) - 1 for node in lines[tour_start:tour_end]]
        
        # Ensure that all indices are within bounds (0 to DIMENSION - 1)
        dimension = len(optimal_tour)
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


    def percentage_of_optimal(solution_distance, optimal_distance):
        if optimal_distance == 0:
            raise ValueError("Optimal distance cannot be zero.")

        percentage = (optimal_distance / solution_distance) * 100
        return percentage


    def initialize_population(number_of_cities):
        population = []
        for _ in range(self.POPULATION_SIZE):
            # population.append(np.random.permutation(number_of_cities))
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


    def tournament_selection(population, distance_matrix):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament.sort(key=lambda x: self.calculate_fitness(x, distance_matrix))  # shouldn't i mark which tours were already used as parents?
        return tournament[0]


    def crossover(parent1, parent2):
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
        

    def mutation(tour):
        new_tour = tour[:]
        # new_tour = tour.copy()
        if random.random() < MUTATION_RATE:
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
        # for i in range(len(tour)):
        #     if random.random() < mutation_rate:
        #         j = random.randint(0, len(tour)-1)
        #         tour[i], tour[j] = tour[j], tour[i]
        # return tour


    def create_new_generation(population, distance_matrix):
        new_population = []
        population_size = len(population)

        for _ in range(population_size):
            parent1 = tournament_selection(population, distance_matrix)
            parent2 = tournament_selection(population, distance_matrix)
            
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        return new_population


    def evolutionary_algorithm(distance_matrix):
        number_of_cities = distance_matrix.shape[0]
        population = initialize_population(number_of_cities)

        best_tour = None
        best_distance = float('inf')

        for generation in range(NUMBER_OF_GENERATIONS):
            population = sorted(population, key=lambda x: calculate_fitness(x, distance_matrix))
            current_best_distance = calculate_fitness(population[0], distance_matrix)

            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_tour = population[0]
            print(f"Generation {generation} - Best Distance: {best_distance}")

            population = create_new_generation(population, distance_matrix)
        return best_tour, best_distance


# def main(problem_name):
#     problem_file_name = problem_name + ".tsp"
#     optimal_tour_file_name = problem_name + ".opt.tour"

#     problem = read_file(problem_file_name)
#     distance_matrix = create_distance_matrix(problem)
#     best_tour, best_distance = evolutionary_algorithm(distance_matrix)
    
#     optimal_tour = read_optimal_tour(optimal_tour_file_name)
#     optimal_distance = calculate_tour_distance(optimal_tour, distance_matrix)
#     percentage = percentage_of_optimal(best_distance, optimal_distance)

#     print("\n- - - - - - - Evolutionary Algorithm - - - - - - -\n")
#     print(f"Best Tour: {best_tour}")
#     print(f"Best Distance: {best_distance}")
#     print("Optimal Tour:", optimal_tour)
#     print("Optimal Distance:", optimal_distance)
#     print("Success ratio: {:.3f}%".format(percentage))


# if __name__ == "__main__":
#     main('bayg29')


