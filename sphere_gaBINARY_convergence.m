function sphere_ga_accurate()
    % Sphere function parameters
    dim = 10;            % Dimension of the Sphere function
    lowerBound = -10;    % Lower bound of the search space
    upperBound = 10;     % Upper bound of the search space

    % Genetic Algorithm Parameters
    populationSize = 500;   % Increased population size for better exploration
    generations = 300;      % Limit to 300 generations
    crossRate = 0.8;        % Crossover rate
    mutateRate = 0.05;      % Mutation rate

    % Initialize population
    candidateSolutions = lowerBound + (upperBound - lowerBound) * rand(populationSize, dim);
    bestFitnessHistory = zeros(1, generations); % Store best fitness per generation

    % Main GA loop
    for gen = 1:generations
        % Evaluate current population
        fitnessScores = arrayfun(@(i) compute_sphere_fitness(candidateSolutions(i, :)), 1:populationSize);

        % Track best fitness
        [bestFitness, bestIndex] = min(fitnessScores);
        bestFitnessHistory(gen) = bestFitness;

        % Adaptive parameters
        adaptiveMutateRate = mutateRate * (1 - gen / generations);
        adaptiveCrossRate = crossRate * (1 - gen / generations);

        % Selection
        selectedParents = perform_selection(candidateSolutions, fitnessScores);

        % Crossover
        offspring = apply_crossover(selectedParents, adaptiveCrossRate);

        % Mutation
        mutatedOffspring = apply_mutation(offspring, adaptiveMutateRate, lowerBound, upperBound);

        % Evaluate new fitness
        newFitness = arrayfun(@(i) compute_sphere_fitness(mutatedOffspring(i, :)), 1:size(mutatedOffspring, 1));

        % Elitism: keep best from previous generation
        candidateSolutions = mutatedOffspring;
        candidateSolutions(1, :) = candidateSolutions(bestIndex, :);  % Retain best solution

        % Select top individuals based on fitness
        combinedFitness = arrayfun(@(i) compute_sphere_fitness(candidateSolutions(i, :)), 1:size(candidateSolutions,1));
        [~, sortedIndexes] = sort(combinedFitness, 'ascend');
        candidateSolutions = candidateSolutions(sortedIndexes(1:populationSize), :);
    end

    % Plot convergence graph
    figure;
    plot(1:generations, bestFitnessHistory, 'LineWidth', 2);
    xlabel('Generation');
    ylabel('Best Fitness');
    title('Convergence Graph for Sphere Function (300 Generations)');
    grid on;
end

% Sphere function (fitness evaluation)
function fitness = compute_sphere_fitness(solution)
    fitness = sum(solution.^2);
end

% Tournament Selection
function selectedParents = perform_selection(population, fitnessValues)
    populationSize = size(population, 1);
    selectedParents = zeros(size(population));

    for i = 1:populationSize
        c1 = randi(populationSize);
        c2 = randi(populationSize);
        if fitnessValues(c1) < fitnessValues(c2)
            selectedParents(i, :) = population(c1, :);
        else
            selectedParents(i, :) = population(c2, :);
        end
    end
end

% Single-point crossover
function offspring = apply_crossover(parents, crossoverProbability)
    popSize = size(parents, 1);
    numGenes = size(parents, 2);
    offspring = parents;

    for i = 1:2:popSize-1
        if rand < crossoverProbability
            crossPoint = randi(numGenes - 1);
            offspring(i, crossPoint+1:end) = parents(i+1, crossPoint+1:end);
            offspring(i+1, crossPoint+1:end) = parents(i, crossPoint+1:end);
        end
    end
end

% Gaussian mutation
function mutatedIndividuals = apply_mutation(offspring, mutationProbability, lowerBound, upperBound)
    mutatedIndividuals = offspring;
    [popSize, numGenes] = size(offspring);

    for i = 1:popSize
        if rand < mutationProbability
            mutationIndex = randi(numGenes);
            mutationValue = offspring(i, mutationIndex) + randn * 0.1;
            mutatedIndividuals(i, mutationIndex) = min(max(mutationValue, lowerBound), upperBound);
        end
    end
end

