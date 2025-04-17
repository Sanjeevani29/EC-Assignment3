function optimize_gap_ga()
    % Create a result file to store the output
    resultFileName = 'gap_ga_results.txt';
    resultFileID = fopen(resultFileName, 'w');
    
    if resultFileID == -1
        error('Error opening result file %s.', resultFileName);
    end
    
    % Write header to the result file
    fprintf(resultFileID, 'Dataset,Instance,TotalBenefit\n');
    
    % Iterate through dataset files (gap1 to gap12)
    for datasetIdx = 1:12
        dataFile = sprintf('gap%d.txt', datasetIdx);
        fileID = fopen(dataFile, 'r');
        if fileID == -1
            error('Error opening file %s.', dataFile);
        end
        
        % Read number of problem instances
        numInstances = fscanf(fileID, '%d', 1);
        
        % Print dataset name (gapX) for display
        fprintf('\n%s\n', dataFile(1:end-4)); % Removes .txt for display
        
        for instanceIdx = 1:numInstances
            % Read problem parameters
            numServers = fscanf(fileID, '%d', 1);
            numUsers = fscanf(fileID, '%d', 1);
            
            % Read cost and resource matrices
            costMatrix = fscanf(fileID, '%d', [numUsers, numServers])';
            resourceMatrix = fscanf(fileID, '%d', [numUsers, numServers])';
            
            % Read server capacities
            capacityLimits = fscanf(fileID, '%d', [numServers, 1]);
            
            % Solve using GA
            assignmentMatrix = execute_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);
            totalBenefit = sum(sum(costMatrix .* assignmentMatrix)); % Maximization
            
            % Print and store results in the file
            fprintf('c%d-%d  %d\n', numServers*100 + numUsers, instanceIdx, round(totalBenefit));
            fprintf(resultFileID, 'gap%d,%d,%d\n', datasetIdx, instanceIdx, round(totalBenefit)); % Save to result file
        end
        
        % Close dataset file
        fclose(fileID);
    end
    
    % Close the result file
    fclose(resultFileID);
    disp('Results saved to gap_ga_results.txt');
end

function assignmentMatrix = execute_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits)
    % Genetic Algorithm Parameters
    populationSize = 100;
    generations = 300;
    crossRate = 0.8;
    mutateRate = 0.02;
    
    % Initialize population
    candidateSolutions = round(rand(populationSize, numServers * numUsers));

    % Enforce feasibility in the initial population
    for idx = 1:populationSize
        candidateSolutions(idx, :) = adjust_feasibility(rand(1, numServers * numUsers), numServers, numUsers);
    end
    
    % Evaluate fitness of initial solutions
    fitnessScores = arrayfun(@(i) compute_fitness(candidateSolutions(i, :)), 1:populationSize);
    
    % Main GA loop
    for gen = 1:generations
        % Selection (Tournament)
        chosenParents = perform_selection(candidateSolutions, fitnessScores);
        
        % Crossover (Single-Point)
        offspring = apply_crossover(chosenParents, crossRate);
        
        % Mutation (Bit Flip)
        mutatedOffspring = apply_mutation(offspring, mutateRate);
        
        % Ensure feasibility
        for i = 1:size(mutatedOffspring, 1)
            mutatedOffspring(i, :) = adjust_feasibility(mutatedOffspring(i, :), numServers, numUsers);
        end
        
        % Evaluate new fitness values
        newFitness = arrayfun(@(i) compute_fitness(mutatedOffspring(i, :)), 1:size(mutatedOffspring, 1));
        
        % Apply elitism
        [~, bestIndex] = max([fitnessScores, newFitness]); 
        if bestIndex > length(fitnessScores)
            candidateSolutions = mutatedOffspring;
            fitnessScores = newFitness;
        else
            candidateSolutions = [candidateSolutions; mutatedOffspring];
            fitnessScores = [fitnessScores, newFitness];
        end
        
        % Select top individuals
        [~, sortedIndexes] = sort(fitnessScores, 'descend');
        candidateSolutions = candidateSolutions(sortedIndexes(1:populationSize), :);
        fitnessScores = fitnessScores(sortedIndexes(1:populationSize));
    end
    
    % Return best solution
    [~, bestIndex] = max(fitnessScores);
    assignmentMatrix = reshape(candidateSolutions(bestIndex, :), [numServers, numUsers]);

    function score = compute_fitness(solution)
        reshapedSolution = reshape(solution, [numServers, numUsers]);
        totalCost = sum(sum(costMatrix .* reshapedSolution)); 
        
        % Apply penalties for constraint violations
        capacityExceedance = sum(max(sum(reshapedSolution .* resourceMatrix, 2) - capacityLimits, 0));
        incorrectAssignment = sum(abs(sum(reshapedSolution, 1) - 1));
        penaltyFactor = 1e6 * (capacityExceedance + incorrectAssignment);
        
        score = totalCost - penaltyFactor;
    end
end

function selectedParents = perform_selection(population, fitnessValues)
    % Tournament Selection
    populationSize = size(population, 1);
    selectedParents = zeros(size(population));
    
    for i = 1:populationSize
        % Pick two candidates randomly
        candidate1 = randi(populationSize);
        candidate2 = randi(populationSize);
        
        % Select the better one
        if fitnessValues(candidate1) > fitnessValues(candidate2)
            selectedParents(i, :) = population(candidate1, :);
        else
            selectedParents(i, :) = population(candidate2, :);
        end
    end
end

function offspring = apply_crossover(parents, crossoverProbability)
    popSize = size(parents, 1);
    numGenes = size(parents, 2);
    offspring = parents;
    
    for i = 1:2:popSize-1
        if rand < crossoverProbability
            % Single-point crossover
            crossPoint = randi(numGenes - 1);
            offspring(i, crossPoint+1:end) = parents(i+1, crossPoint+1:end);
            offspring(i+1, crossPoint+1:end) = parents(i, crossPoint+1:end);
        end
    end
end

function mutatedIndividuals = apply_mutation(offspring, mutationProbability)
    mutatedIndividuals = offspring;
    for i = 1:numel(offspring)
        if rand < mutationProbability
            mutatedIndividuals(i) = 1 - mutatedIndividuals(i);
        end
    end
end

function feasibleSolution = adjust_feasibility(solution, numServers, numUsers)
    % Ensure each user is assigned exactly once
    reshapedSol = reshape(solution, [numServers, numUsers]);
    for j = 1:numUsers
        [~, idx] = max(reshapedSol(:, j));
        reshapedSol(:, j) = 0;
        reshapedSol(idx, j) = 1;
    end
    feasibleSolution = reshape(reshapedSol, [1, numServers * numUsers]);
end
