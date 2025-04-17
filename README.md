This code solves the Generalized Assignment Problem (GAP) using a Genetic Algorithm (GA).
The datasets named gap1.txt to gap12.txt, each containing multiple problem instances. For every instance, 
it reads the number of servers and users, along with their cost, resource requirements, and server capacity constraints.
 
The GA is designed to maximize the total benefit by finding an optimal assignment of users to servers while respecting the constraints.
It initializes a population, applies selection, crossover, and mutation, and corrects infeasible solutions to ensure valid assignments.

Fitness is calculated by combining cost and penalties for constraint violations. The best solution from each instance is reshaped into an assignment matrix, 
and the total benefit is computed. Results for each dataset and instance are printed to the console and stored in a file named gap_ga_results.txt
