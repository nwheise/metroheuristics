import numpy as np
import pandas as pd
import time

class AntGraph():
    def __init__(self, vertices, edges, max_time):
        '''
        Parameters: vertices -> numpy Array with two columns, contains index and
                                vertex name
                    edges -> numpy Array of size [i,j], where element [i,j] is
                             the edge length between vertex i and j
                    max_time -> Int, maximum time for Ant to travel
        '''
        self.edges = edges
        self.max_time = max_time

        # Initialize pheromones to 1 on all edges. Like edges, pheromones is size
        # [i,j] where [i,j]th element is the pheromones on edge from i to j
        self.pheromones = np.ones((edges.shape))

        # Create mapping from indices to names
        self.index_mapping = {int(v[0]): v[1] for v in vertices}

        # Parameters for movement probabilities... alpha is weight of pheromones
        # and beta is weight of edge lengths
        self._alpha = 1
        self._beta = 1.5

        self.best_ever = Ant(start_vertex=0)


    def create_ants(self, n):
        '''
        Parameters: n -> Int
        Create n ants with random starting vertices
        '''
        start_vertices = np.random.randint(low=0, high=self.edges.shape[0], size=n)
        self.ants = []
        for v in start_vertices:
            self.ants.append(Ant(start_vertex=130))
        self.ants = np.array(self.ants)


    def probability_of_moving(self, ant, vertex_i, vertex_j):
        '''
        Parameters: ant -> Ant
                    vertex_i -> Int, index of current vertex
                    vertex_j -> Int, index of potential next vertex
        Calculate the probability of moving from i to j
        '''
        # Traditional equation variables
        tau = np.power(self.pheromones[vertex_i, vertex_j], self._alpha)
        eta = np.power(np.reciprocal(self.edges[vertex_i, vertex_j]), self._beta)
        denom = np.sum(self._denom[vertex_i])

        # My modification, to give more weight to unvisited vertices
        neighbors = np.where(np.isfinite(metro.edges[vertex_i]))
        total_neighbor_visits = 0
        for neighbor_index in neighbors[0]:
            if neighbor_index not in ant.visits:
                ant.visits[neighbor_index] = 0
            total_neighbor_visits += 1 / (ant.visits[neighbor_index] + 1)

        visited_coeff = (1 / (ant.visits[vertex_j] + 1)) / total_neighbor_visits

        return (visited_coeff + (tau*eta/denom)) / 2


    def move_ants(self):
        '''
        Move all ants based on their movement probabilities until each one has
        travelled for max_time.
        '''
        for ant in self.ants:
            if not ant.done_moving(max_time=self.max_time, mapping=self.index_mapping):
                current_v = ant.current_vertex()
                rand_val = np.random.random()
                prob = 0
                neighbors = np.where(np.isfinite(metro.edges[current_v]))
                for v in neighbors[0]:
                    prob += self.probability_of_moving(ant=ant, vertex_i=current_v, vertex_j=v)
                    if prob > rand_val:
                        ant.move_to(new_vertex=v, time=self.edges[current_v, v])
                        break


    def ants_still_moving(self):
        '''
        Return True if every Ant has time left to travel and hasn't reached all
        unique vertices
        '''
        for ant in self.ants:
            if not ant.done_moving(max_time=self.max_time, mapping=self.index_mapping):
                return True
        return False


    def leave_pheromones(self):
        '''
        Update the pheromones left by Ants
        '''
        # evaporate previous pheromones
        rho = 0.25 # pheromone evaporation coefficient, in range [0, 1]
        self.pheromones = np.multiply(self.pheromones, (1 - rho))

        # add new pheromones if ants travelled that edge, with stronger pheromones
        # if the Ant had a good score (visited many unique vertices)
        for ant in self.ants:
            for i in range(self.edges.shape[0]):
                for j in range(self.edges.shape[1]):
                    if ant.travelled_edge(edge=[i, j]):
                        self.pheromones[i, j] += ant.path_score(mapping=self.index_mapping)


    def update_denom(self):
        '''
        For faster computation, calculate denom for probability_of_moving once
        per iteration
        '''
        self._denom = np.multiply(np.power(self.pheromones, self._alpha), np.power(np.reciprocal(self.edges), self._beta))


    def run_iteration(self, num_ants):
        '''
        Parameters: num_ants -> Int, number of Ants to create and let travel
        Run one iteration of the algorithm
        '''
        self.create_ants(n=num_ants)
        self.update_denom()
        while self.ants_still_moving():
            self.move_ants()
        self.leave_pheromones()
        self.update_best_ever()


    def update_best_ever(self):
        '''
        Keep track of the best Ant
        '''
        for ant in self.ants:
            if ant.path_score(mapping=self.index_mapping) > self.best_ever.path_score(mapping=self.index_mapping):
                self.best_ever = ant


class Ant():
    def __init__(self, start_vertex):
        '''
        Parameters: start_vertex -> Int, index of starting vertex
        Initialize Ant. Path is stored as a list of two vertices, indicating
        that the Ant travelled from the first to the second. Visits to vertices 
        are stored as a dictionary.
        '''
        self.path = np.array([[int(start_vertex), int(start_vertex)]])
        self.visits = {int(start_vertex): 1}
        self.time = 0


    def __str__(self):
        return f'(V: {self.current_vertex()}, T: {self.time})'


    def __repr__(self):
        return self.__str__()


    def current_vertex(self):
        '''
        Return most recent visited vertex
        '''
        return self.path[-1, 1]


    def move_to(self, new_vertex, time):
        '''
        Parameters: new_vertex -> Int, vertex index to add to end of path
                    time -> Int, time needed to move to the new vertex
        Append [current vertex, next vertex] to path to effectively move the Ant
        to a new vertex. Update visit counts and time accordingly.
        '''
        self.path = np.append(arr=self.path, values=[[self.current_vertex(), int(new_vertex)]], axis=0)
        if new_vertex in self.visits:
            self.visits[new_vertex] += 1
        else:
            self.visits[new_vertex] = 0
        self.time += int(time)


    def done_moving(self, max_time, mapping):
        '''
        Parameters: max_time -> Int, maximum time allowed to travel
                    mapping -> Dict, keys are indices and values are names as
                               Strings to convert from indices to names
        Return True if the Ant has travelled for the max_time, or if all 
        vertices with unique names have been visited at least once.
        '''
        visited = self.count_unique_stations(mapping=mapping)
        total_stations = len(set(mapping.values()))
        if (self.time >= max_time) or (visited == total_stations):
            return True
        return False


    def travelled_edge(self, edge):
        '''
        Parameters: edge -> list [i, j], corresponding to edge from vertex i to j
        Checks if the edge is in the Ant's path.
        '''
        if edge[0] == edge[1]:
            return False
        return np.equal(edge, self.path).all(axis=1).any()


    def visited_vertex(self, vertex_index):
        '''
        Parameters: vertex_index -> Int
        Returns True if vertex with index vertex_index has been visited
        '''
        if (vertex_index not in self.visits) or (self.visits[vertex_index] == 0):
            return False
        return True


    def count_unique_stations(self, mapping):
        '''
        Parameters: mapping -> Dict, keys are indices and values are names as
                               Strings to convert from indices to names
        Counts number of vertices with unique names that are in the path.
        '''
        names = []
        for index in self.path[:, 0]:
            if mapping[index] not in names:
                names.append(mapping[index])
        return len(names)


    def path_score(self, mapping):
        '''
        Parameters: mapping -> Dict, keys are indices and values are names as
                               Strings to convert from indices to names
        Returns path score as ratio of vertices with unique names in path over 
        total possible vertices with unique names, such that 1 is returned if all
        vertices with unique names are visited.
        '''
        score = self.count_unique_stations(mapping=mapping) / len(set(mapping.values()))
        return score


if __name__ == '__main__':
    # Read the vertex data and convert to numpy array
    stations = pd.read_csv('stations.csv', sep=';', header=None, encoding='latin1')
    stations = np.array(stations)

    # Read the edge data and convert to numpy array
    edges_df = pd.read_csv('edges.csv', sep=';', header=None)
    num_vertices = edges_df[0].max() + 1
    edges = np.full(shape=(num_vertices, num_vertices), fill_value=np.inf)
    for row in edges_df.itertuples():
        edges[row[1], row[2]] = row[3]


    # Create metro as AntGraph object
    metro = AntGraph(vertices=stations, edges=edges, max_time=72000)
    index_mapping = {int(v[0]): v[1] for v in stations} # mapping from indices to names

    # Run iterations
    for i in range(50):
        t0 = time.time()
        metro.run_iteration(num_ants=50)
        t1 = time.time()

        print(f'Iteration {i} complete in {t1-t0} seconds.')
        print(f'Best so far: {metro.best_ever.count_unique_stations(mapping=index_mapping)} unique stations')

    # Print best path found and the path score
    print('Best Path:')
    print(metro.best_ever.path)
    print('Fraction of Unique Stations Visited:')
    print(metro.best_ever.count_unique_stations(mapping=index_mapping))

    # Save the best path to a csv file
    np.savetxt(fname='aco_best_path.csv', X=metro.best_ever.path, fmt='%d', delimiter=',')
