function gap12_convergence()
    % Read GAP12 dataset (only the first instance)
    dataFile = 'gap12.txt';
    fileID = fopen(dataFile, 'r');
    if fileID == -1
        error('Cannot open file: %s', dataFile);
    end

    numInstances = fscanf(fileID, '%d', 1);

    % Read only the first instance
    numServers = fscanf(fileID, '%d', 1);
    numUsers = fscanf(fileID, '%d', 1);
    costMatrix = fscanf(fileID, '%d', [numUsers, numServers])';
    resourceMatrix = fscanf(fileID, '%d', [numUsers, numServers])';
    capacityLimits = fscanf(fileID, '%d', [numServers, 1]);
    fclose(fileID);

    [~, convergence] = run_ga_gap12(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

    % Plot convergence
    figure;
    plot(convergence, 'LineWidth', 2);
    title('GA Convergence - GAP12 (Instance 1)');
    xlabel('Generation');
    ylabel('Best Fitness');
    grid on;
end

function [bestSol, convergence] = run_ga_gap12(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits)
    popSize = 100;
    generations = 100; % Changed from 300 to 100
    crossRate = 0.8;
    mutateRate = 0.02;

    % Initial population
    pop = randi([0, 1], popSize, numServers * numUsers);
    for i = 1:popSize
        pop(i,:) = enforce_feasibility(pop(i,:), numServers, numUsers);
    end

    fitness = evaluate_population(pop, numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);
    convergence = zeros(generations, 1);

    for gen = 1:generations
        parents = tournament_selection(pop, fitness);
        offspring = single_point_crossover(parents, crossRate);
        offspring = bitflip_mutation(offspring, mutateRate);

        for i = 1:popSize
            offspring(i,:) = enforce_feasibility(offspring(i,:), numServers, numUsers);
        end

        offspringFitness = evaluate_population(offspring, numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

        [pop, fitness] = elitism_selection([pop; offspring], [fitness; offspringFitness], popSize);
        convergence(gen) = max(fitness);
    end

    [~, bestIdx] = max(fitness);
    bestSol = reshape(pop(bestIdx, :), [numServers, numUsers]);
end

function fitness = evaluate_population(pop, m, n, costMatrix, resourceMatrix, capacityLimits)
    fitness = zeros(size(pop,1),1);
    for i = 1:size(pop,1)
        sol = reshape(pop(i,:), [m, n]);
        benefit = sum(sum(sol .* costMatrix));
        violation = sum(max(sum(sol .* resourceMatrix, 2) - capacityLimits, 0));
        assignmentError = sum(abs(sum(sol,1) - 1));
        penalty = 1e6 * (violation + assignmentError);
        fitness(i) = benefit - penalty;
    end
end

function newPop = enforce_feasibility(sol, m, n)
    mat = reshape(sol, [m, n]);
    for j = 1:n
        [~, idx] = max(mat(:, j));
        mat(:, j) = 0;
        mat(idx, j) = 1;
    end
    newPop = reshape(mat, [1, m * n]);
end

function selected = tournament_selection(pop, fitness)
    popSize = size(pop, 1);
    selected = zeros(size(pop));
    for i = 1:popSize
        a = randi(popSize);
        b = randi(popSize);
        if fitness(a) > fitness(b)
            selected(i,:) = pop(a,:);
        else
            selected(i,:) = pop(b,:);
        end
    end
end

function offspring = single_point_crossover(parents, rate)
    [popSize, numGenes] = size(parents);
    offspring = parents;
    for i = 1:2:popSize-1
        if rand < rate
            point = randi(numGenes-1);
            offspring(i,point+1:end) = parents(i+1,point+1:end);
            offspring(i+1,point+1:end) = parents(i,point+1:end);
        end
    end
end

function mutated = bitflip_mutation(pop, rate)
    mutated = pop;
    for i = 1:numel(pop)
        if rand < rate
            mutated(i) = 1 - pop(i);
        end
    end
end

function [newPop, newFit] = elitism_selection(pop, fit, maxSize)
    [fitSorted, idx] = sort(fit, 'descend');
    newPop = pop(idx(1:maxSize), :);
    newFit = fitSorted(1:maxSize);
end
