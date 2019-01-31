# metroheuristics
<p>Implementations of metaheuristic algorithms, applied to exploring the Paris Metro</p>

## Description
<p>The overall objective of this project is to use metaheuristics to find a path that explores 
all Paris metro stations within one day. A secondary objective is, if all stations 
are visited, to find the shortest path that still visits all stations. In this case, we consider one day to be from 5am to 1am
(20 hours, or 72000 seconds). We have been provided with data for stations and edges
between them. Notably, each station has an independent index for each metro line that
passes through it (e.g. indices 0311 through 0315 are all station République since
there are five lines that pass through République). We do not need to visit each
index, only each unique station.</p>

<p>The implementations here are generalizable to any graph with vertices, 
provided that they are given in the correct format. The edges.txt and stations.txt
files in this repo are specific to the Paris Metro system.</p>

## Approach
<p>Three approaches are implemented here: A greedy solution, ant colony optimization, and a genetic aglorithm.</p>
Ant Colony Optimization (ACO)
<p>Two classes are used: an AntGraph and Ant. An initial number of ants are allowed
to explore the AntGraph. Once they have been allowed to travel a complete
path (reaching max time or visiting all stations), they are removed and leave
pheromones on the edges taken as a function of their score - in this case, how
many unique stations were visited. New populations of Ants are released to explore
the AntGraph, and have an increased probability to follow trails with high
pheromones. Over time, Ants aggregate to the same path which is ideally a good 
approximation of the optimal solution.</p>

<p>In this case, ACO requires high computation time and yields relatively poor results.</p>

Greedy Solution
<p>In the case of the Paris Metro, a greedy solution is quite well suited to find a valid solution
in relatively low time. The greedy solution here means that, at a given vertex, the next vertex
is chosen as the neighbor with the fewest visits so far (to reduce re-traveling to the same
vertices). In the case of a tie, the shortest edge is chosen. If there are multiple neighbors with
equal minimal visits and equally long edges between them, one is chosen arbitrarily.</p>

Genetic Algorithm
<p>The genetic algorithm here is a simple genetic algorithm which uses generations of individuals to
iteratively produce better populations. Generation 0 Individuals initially explore the graph,
and then are given a fitness score based on their time and number of unique vertices visited.
Selection, crossover, and mutation are performed to produce new generations until a max number
of generations are produced.</p>

## Files
<p>The files included in this repo are:</p>
<ul>
  <li>stations.txt : text file containing each station (vertex) as a line, 
  where each has an index and name</li>
  <li>edges.txt : text file containing each edge as a line in the form 'i j d', 
  where i is the start vertex index, j is the end vertex index, 
  and d is the distance between them</li>
  <li>clean-data.py : script used to convert raw data of the same form as edges.txt 
  and stations.txt into csv files usable by this implementation</li>
  <li>metaheuristics.py : file containing class implementations of the 
  greedy algorithm and genetic algorithm, to be imported and used by 
  other scripts</li>
  <li>genetic-soln.py : script to run and output results for genetic algorithm
  solution to the Paris Metro exploration problem</li>
  <li>greedy-soln.py : script to run and output results for greedy algorithm
  solution to the Paris Metro exploration problem</li>
  <li>ant-colony-soln.py : script to run for ant colony algorithm solution
  to the Paris Metro exploration problem (note, ant-colony-soln.py is 
  independent of the metaheuristics.py classes)</li>
  <li>requirements.txt : necessary modules</li>
</ul>
