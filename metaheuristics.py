import pandas as pd
import numpy as np
import time
import copy
import random


class Graph():
    '''
    Generic class for graphs, implemented by a list of Vertex objects.
    '''
    def __init__(self, vertices_file, edges_file):
        self.vertices = []
        self.vertices_from_csv(filename=vertices_file)
        self.generate_edges(filename=edges_file)


    def __getitem__(self, index):
        return self.vertices[index]


    def __len__(self):
        return len(self.vertices)


    def vertices_from_csv(self, filename):
        '''
        Parameters: filename -> String, file name with type '.csv'
        Reads csv file and creates all Vertices with index and name.
        '''
        vertices_df = pd.read_csv(filename, sep=';', header=None, encoding='latin1')
        for row in vertices_df.itertuples():
            self.vertices.append(Vertex(index=row[1], name=row[2]))


    def generate_edges(self, filename):
        '''
        Parameters: filename -> String, file name with type '.csv'
        Assignes edges to each Vertex by reading from the csv file
        '''
        for v in self.vertices:
            v.edges_from_csv(filename)


    def count_unique_vertices(self):
        '''
        Returns count of vertices with unique names, rather than unique indexes
        '''
        names = [v.name for v in self.vertices]
        return len(set(names))


class Vertex():
    '''
    Vertex class with attributes index, name, a list of Edge objects, and number
    of visits
    '''
    def __init__(self, index, name):
        '''
        Parameters: index -> Int
                    name -> String
        Initialize Vertex with index, name, empty list of Edge objects, and visits
        '''

        self.index = int(index)
        self.name = name
        self.edges = []
        self.visits = 0


    def edges_from_csv(self, filename):
        '''
        Parameters: filename -> String, file name with type '.csv'
        Generate list of edges attribute for this vertex by reading from input csv
        '''
        edges_df = pd.read_csv(filename, sep=';', header=None)
        for row in edges_df.itertuples():
            if row[1] == self.index:
                self.edges.append(Edge(index_i=row[1], index_j=row[2], distance=row[3]))


    def neighbor_indices(self):
        '''
        Returns list of neighbor indices
        '''
        neighbors = []
        for e in self.edges:
            neighbors.append(e.index_j)
        return neighbors


    def matches_name(self, v):
        '''
        Parameters: v -> Vertex
        Takes Vertex object v and compares the name strings
        '''
        return (self.name == v.name)


class Edge():
    '''
    Edge object to connect two Vertex objects. Contains only index_i and index_j,
    to designate the edge goes from i to j, and the distance.
    '''
    def __init__(self, index_i, index_j, distance):
        '''
        Parameters: index_i -> Int, starting Vertex index
                    index_j -> Int, ending Vertex index
                    distance -> Int, distance between the Vertices
        '''

        self.index_i = index_i
        self.index_j = index_j
        self.distance = distance


    def __str__(self):
        return f'{self.index_i}:{self.index_j}:{self.distance}'


    def __repr__(self):
        return self.__str__()


class GreedyAlgorithm():
    '''
    Greedy algorithm to find a path containing all stations with unique names.
    Path is stored as a list of integers, corresponding to Vertex indexes
    '''
    def __init__(self, graph):
        '''
        Parameters: graph -> Graph
        Initialize GreedyAlgorithm with empty path
        '''
        self.graph = copy.deepcopy(graph)
        self.path = []


    def set_start(self, v):
        '''
        Parameters: v -> Int
        Initialize path with start index v
        '''
        self.path = [v]
        self.graph[v].visits += 1


    def get_next(self):
        '''
        Selects best neighbor of current vertex. This first chooses the neighbors
        with least visits. If there is a tie (i.e. multiple neighbors have been
        visited the same number of times), then pick the shortest edge.
        '''
        current_idx = self.path[-1]
        current_v = self.graph[current_idx]
        neighbor_idxs = [edge.index_j for edge in current_v.edges]
        neighbor_visits = [self.graph[idx].visits for idx in neighbor_idxs]

        min_visits = min(neighbor_visits)
        least_visited_neighbors = [neighbor_idxs[idx] for idx, visits in enumerate(neighbor_visits) if visits == min_visits]

        neighbor_distances = [edge.distance for edge in current_v.edges if edge.index_j in least_visited_neighbors]
        min_distance = min(neighbor_distances)
        best_neighbors = [(least_visited_neighbors[idx], distance) for idx, distance in enumerate(neighbor_distances) if distance == min_distance]

        return best_neighbors[0]


    def run(self, max_time):
        '''
        Parameters: max_time -> Int
        Continues to add to the path as long as max time has not been exceeded.
        If 100% of unique stations have been visited, stops.
        '''
        self._t = 0
        while (self._t < max_time) and (self.fraction_visited() < 1):
            next_v = self.get_next()
            self.path.append(next_v[0])
            self.graph[next_v[0]].visits += 1
            self._t += next_v[1]


    def get_runtime(self):
        '''
        Returns the path time
        '''
        return self._t


    def fraction_visited(self):
        '''
        Counts Vertices with unique names in path. Returns a ratio of unique
        Vertices in path over total possible unique Vertices, such that 1 is
        returned if all unique Vertices are visited.
        '''
        visited_names = [v.name for v in self.graph if v.index in self.path]
        num_unique_names = len(set(visited_names))
        return num_unique_names / self.graph.count_unique_vertices()


    def path_to_string(self):
        '''
        Returns path list with index integers converted to name strings.
        '''
        path_names = [self.graph[idx].name for idx in self.path]
        return path_names


class Individual():
    '''
    Class used within GeneticAlgorithm class. An Individual has its own path
    along the graph and a corresponding path_score and time_score based on 
    unique visited Vertices and total path time, respectively.
    '''
    def __init__(self, graph):
        '''
        Parameters: graph -> Graph
        Initialize Individual with empty path, set path score to 0 and time score
        to infinity
        '''
        self.graph = copy.deepcopy(graph)
        self.path = []
        self.path_score = 0
        self.time_score = np.inf


    def __str__(self):
        '''
        Prints PS (path score) and TS (time score) for the Individual
        '''
        return f'(PS: {self.path_score}, TS: {self.time_score})'


    def __repr__(self):
        return self.__str__()


    def set_start(self, v):
        '''
        Parameters: v -> Int
        Intializes path with start Vertex index v. If v = -1, chooses random
        starting Vertex.
        '''
        if v == -1:
            v = np.random.randint(low=0, high=len(self.graph))
        self.path = [v]
        self.graph[v].visits += 1


    def random_choice(self):
        '''
        Randomly chooses a neighbor from the current Vertex.
        Returns tuple of (chosen neighbor index, edge distance)
        '''
        current_idx = self.path[-1]
        current_v = self.graph[current_idx]

        choice_idx = np.random.randint(low=0, high=len(current_v.edges))
        choice_edge = current_v.edges[choice_idx]

        return (choice_edge.index_j, choice_edge.distance)


    def probabilistic_choice(self, criteria):
        '''
        Parameters: criteria -> 'dist', 'visits', or 'both', 
        Chooses neighbor probabilistically, selecting neighbor based on inverse 
        of distance and/or inverse of visits relative to other neighbors.
        Returns tuple of (chosen neighbor index, edge distance)
        '''
        current_idx = self.path[-1]
        current_v = self.graph[current_idx]

        # Calculate denominators (sum of 1/distance, sum of 1/(visits+1) for all neighbors)
        dist_denom = 0
        visits_denom = 0
        for e in current_v.edges:
            neighbor_idx = e.index_j
            dist_denom += (1 / e.distance)
            visits_denom += (1 / (self.graph[neighbor_idx].visits + 1))

        # Calculate probabilities as (1/distance_denominator + 1/visits_denominator)/2
        prob_list = []
        for e in current_v.edges:
            neighbor_idx = e.index_j
            dist_prob = (1 / e.distance) / dist_denom
            visits_prob = (1 / (self.graph[neighbor_idx].visits + 1)) / visits_denom

            choice_prob = (dist_prob + visits_prob) / 2

            if criteria == 'dist':
                prob_list.append(dist_prob)
            elif criteria == 'visits':
                prob_list.append(visits_prob)
            elif criteria == 'both':
                prob_list.append(choice_prob)
            else:
                raise Exception(f'Invalid parameter for "criteria": {criteria}')

        # Select neighbor based on these probabilities.
        rand = np.random.random()
        prob_sum = 0
        for i in range(len(prob_list)):
            prob_sum += prob_list[i]
            if prob_sum >= rand:
                return (current_v.edges[i].index_j, current_v.edges[i].distance)


    def explore(self, max_time, choice):
        '''
        Parameters: max_time -> Int
                    choice -> String, 'probabilistic' or 'random'
        Lets individual travel along graph, choosing it's path depending on the
        choice parameter. Continues until max_time is reached.
        '''
        t = 0
        while (t < max_time) and (self.fraction_visited() < 1):
            if choice == 'probabilistic':
                chosen_edge = self.probabilistic_choice(criteria='visits')
            elif choice == 'random':
                chosen_edge = self.random_choice()
            else:
                raise Exception(f'Invalid choice parameter for Individual.explore(): {choice}')
            self.path.append(chosen_edge[0])
            self.graph[chosen_edge[0]].visits += 1
            t += chosen_edge[1]
        self.score_path()


    def fraction_visited(self):
        '''
        Counts Vertices with unique names in path. Returns a ratio of unique
        Vertices in path over total possible unique Vertices, such that 1 is
        returned if all unique Vertices are visited.
        '''

        visited_names = [v.name for v in self.graph if v.index in self.path]
        num_unique_names = len(set(visited_names))
        return num_unique_names / self.graph.count_unique_vertices()


    def score_path(self):
        '''
        Assigns a path_score as the fraction of unique Vertices visited over the
        total possible. Assigns a time_score as the time taken to travel the path
        '''
        self.time_score = 0
        self.path_score = 0
        for i in range(len(self.path) - 1):
            v_idx = self.path[i]
            for e in self.graph[v_idx].edges:
                if e.index_j == self.path[i+1]:
                    self.time_score += e.distance
        self.path_score = self.fraction_visited()


class GeneticAlgorithm():
    '''
    Class which uses a list of Individual objects to simulate a population.
    Individuals are selected based on fitness, then crossover and mutate to 
    produce new generations.
    '''
    def __init__(self, graph):
        '''
        Parameters: graph -> Graph
        Initialize GeneticAlgorithm with no Individuals in population
        '''
        self.graph = copy.deepcopy(graph)
        self.population = []

        # Champion to keep track of the best performing Individual ever seen
        self.champion = Individual(self.graph)

    
    def create_population(self, pop_size, max_time):
        '''
        Parameters: pop_size -> Int
                    max_time -> Int
        Intializes population of pop_size Individuals. Each Individual explores 
        the graph for max_time seconds.
        '''
        
        self.population = [Individual(graph=self.graph) for i in range(pop_size)]
        i = 1
        for indiv in self.population:
            indiv.set_start(v=-1)
            indiv.explore(max_time=max_time, choice='probabilistic')
            print(f'Individual {i}/{pop_size} created in initial population.')
            i += 1
        self.pick_champion()


    def pick_champion(self):
        '''
        Selects the best Individual seen throughout the generations. 
        Path score is prioritized, but if path score is equal then the 
        Individual with a lower time score is chosen.
        '''
        for indiv in self.population:
            if indiv.path_score > self.champion.path_score:
                self.champion = indiv
            elif indiv.path_score == self.champion.path_score:
                if indiv.time_score < self.champion.time_score:
                    self.champion = indiv


    def selection(self, kind='tournament', trunc_proportion=0.5, tourn_size=4, tourn_prob=0.4):
        '''
        Parameters: kind -> String, 'tournament' or 'truncation'
                    trunc_proportion -> Float [0,1), only used when kind='truncation',
                                        proportion of population truncated
                    tourn_size -> Int, only used when kind='tournament', 
                                  num of individuals chosen in tournaments
                    tourn_prob -> Float [0,1), only used when kind='tournament',
                                  most fit Individual is chosen from tournament
                                  with probability tourn_prob, 2nd most fit with
                                  probability tourn_prob*(1-tourn_prob), third 
                                  most fit with probability tourn_prob(1-tourn_prob)**2, etc
        Selects some subpopulation of the population based on fitness (measured
        by path score and time score). When kind='truncation', a certain percentage
        of Individuals are dropped. When kind='tournament', tournament selection
        is performed.
        '''

        if kind == 'truncation':
            self.population.sort(key=lambda individual: (1-individual.path_score, individual.time_score))
            new_pop_size = int(trunc_proportion*len(self.population))
            self.population = self.population[:new_pop_size]     
        elif kind == 'tournament':
            tournament = [self.population[i:i+tourn_size] for i in range(0, len(self.population), tourn_size)]
            self.population = []
            for sub_pop in tournament:
                sub_pop.sort(key=lambda individual: (1-individual.path_score, individual.time_score))
                rand = np.random.random()
                prob_sum = tourn_prob
                for indiv in sub_pop:
                    if rand < prob_sum:
                        self.population.append(indiv)
                        break
                    prob_sum += tourn_prob*(1-tourn_prob)
        else:
            raise Exception(f'Invalid selection parameter: {kind}')   


    def crossover(self, end_pop_size=100, max_time=72000):
        '''
        Parameters: end_pop_size -> Int, desired population size
                    max_time -> Int, used to ensure child's path is still within
                                allowable time
        Performs single point crossover to produce children from selected
        Individuals in population.
        Individuals are chosen randomly, and a crossover point is chosen randomly
        from one of their paths. The second part of the path, beginning at the
        crossover point, is swapped with the second part of the other parent's
        path to create a new child Individual with elements of both parents' paths.
        '''

        new_pop = self.population
        while len(new_pop) < end_pop_size:
            child = Individual(graph=self.graph)
            parent_a = random.choice(self.population)
            parent_b = random.choice(self.population)
            while parent_a == parent_b:
                parent_b = random.choice(self.population)

            attempts = 0
            while attempts < 5:
                try:
                    crossover_point = random.choice(parent_a.path)
                    co_idx_a = parent_a.path.index(crossover_point)
                    co_idx_b = parent_b.path.index(crossover_point)
                except ValueError:
                    attempts += 1
                    continue
                break

            if attempts == 5:
                break

            child.path = parent_a.path[:co_idx_a] + parent_b.path[co_idx_b:]
            child.score_path()

            if child.time_score <= max_time:
                new_pop.append(child)
        self.population = new_pop


    def mutation(self, m_prob=0.05, mut_path_time=1000, max_time=72000):
        '''
        Parameters: m_prob -> Float [0,1], probability for an Individual to mutate
                    mut_path_time -> Int, length of mutated path to be inserted
                    max_time -> Int, used to ensure new mutated path is still
                                within allowable time
        Mutates the population. Each Individual has a m_prob probability to have
        their path mutated. A random point on the Individual's path is chosen,
        and a new path with path time of mut_path_time is generated. The end
        Vertex on the mutation path is found on the original path, and the mutation
        path is inserted so that the start and end Vertices connect with the
        original path.
        '''
        for indiv in self.population:
            rand = np.random.random()
            if rand < m_prob:
                mutated = False
                while not mutated:
                    mut = Individual(graph=self.graph)
                    while True:
                        try:
                            mut_point = random.choice(indiv.path)
                            mut.set_start(mut_point)
                            mut.explore(max_time=mut_path_time, choice='probabilistic')
                            end_point = mut.path[-1]

                            start_idx = indiv.path.index(mut_point)
                            end_idx = indiv.path.index(end_point)
                        except ValueError:
                            continue
                        break

                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx

                    first_path_splice = indiv.path[:start_idx]
                    mutated_path = mut.path
                    second_path_splice = indiv.path[end_idx:]

                    original_path = indiv.path
                    indiv.path = first_path_splice + mutated_path + second_path_splice
                    indiv.score_path()

                    if indiv.time_score <= max_time:
                        mutated = True
                    else:
                        indiv.path = original_path
                        indiv.score_path()


    def champion_path_to_string(self):
        '''
        Returns list of Vertex names in the champion's path
        '''
        path_names = [self.graph[idx].name for idx in self.champion.path]
        return path_names