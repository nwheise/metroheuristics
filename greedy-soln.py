import pandas as pd
import metaheuristics as mh
import time
if __name__ == '__main__':
    # Iterate over all 376 starting Vertices for the metro
    rows_list = []
    for i in range(376):
        t0 = time.time()

        # Read metro data in as Graph object
        metro = mh.Graph(vertices_file='stations.csv', edges_file='edges.csv')

        # Intialize GreedyAlgorithm with start Vertex index i
        ga = mh.GreedyAlgorithm(graph=metro)
        ga.set_start(v=i)

        # Run the algorithm
        ga.run(max_time=72000)

        # Save how the algorithm performed with this start location to a list
        temp_dict = {'startVertex': i, 'fractionVisited': ga.fraction_visited(), 'time': ga.get_runtime()}
        rows_list.append(temp_dict)

        t1 = time.time()
        print(f'Test w/ start vertex {i} complete in {t1-t0} seconds')

    # Write the results of each starting position to a csv file
    ga_results = pd.DataFrame(rows_list)
    ga_results.to_csv('ga_results.csv')

    # Choose the starting Vertex where the path was shortest
    best_start_v = ga_results.loc[ga_results['time'].idxmin(), 'startVertex']

    # Rerun only this solution to obtain the best path
    metro = mh.Graph(vertices_file='stations.csv', edges_file='edges.csv')
    ga = mh.GreedyAlgorithm(graph=metro)
    ga.set_start(v=best_start_v)
    ga.run(max_time=72000)

    # Save the best path to a csv file
    path_df = pd.DataFrame({'index': ga.path, 'name': ga.path_to_string()})
    path_df.to_csv('greedy_best_path.csv')