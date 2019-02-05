import pandas as pd
import matplotlib.pyplot as plt
import metaheuristics as mh
import time


if __name__ == '__main__':
    # Read metro data in as Graph object
    metro = mh.Graph(vertices_file='stations.csv', edges_file='edges.csv')

    # Initialize GeneticAlgorithm with first population
    gen_alg = mh.GeneticAlgorithm(graph=metro)
    t0 = time.time()
    gen_alg.create_population(pop_size=100, max_time=72000)
    t1 = time.time()
    print(f'Initial population created in {t1-t0} seconds.')

    # Run algorithm
    num_generations = 2500
    best_path_scores = []
    best_time_scores = []
    for i in range(num_generations):
        # Each generation, perform selection, crossover, and mutation
        t0 = time.time()
        gen_alg.selection(kind='tournament', tourn_size=4, tourn_prob=0.5)
        gen_alg.crossover(end_pop_size=100)
        gen_alg.mutation(m_prob=0.1, mut_path_time=1440)

        # Update the champion
        gen_alg.pick_champion()
        t1 = time.time()

        # Append best scores seen to list, to be plotted
        best_path_scores.append(gen_alg.champion.path_score)
        best_time_scores.append(gen_alg.champion.time_score)
        print(f'Generation {i} complete in {t1-t0} seconds.')
        print(f'Best ever: {gen_alg.champion}')

    # Plot scores over generations
    plt.plot(range(num_generations), best_path_scores)
    plt.xlabel(xlabel='Generation')
    plt.ylabel(ylabel='Percent Stations Visited')
    plt.title(label='Percent of Visited Stations by Best Individual, by Generation')
    plt.savefig('genetic_best_path_per_generation.png')
    plt.clf()

    plt.plot(range(num_generations), best_time_scores)
    plt.xlabel(xlabel='Generation')
    plt.ylabel(ylabel='Time to Travel Best Path')
    plt.title(label='Time of Best Path, by Generation')
    plt.savefig('genetic_best_time_per_generation.png')

    # Write the best path to a csv file
    path_df = pd.DataFrame({'index': gen_alg.champion.path, 'name': gen_alg.champion_path_to_string()})
    path_df.to_csv('genetic_best_path.csv')