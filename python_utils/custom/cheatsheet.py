"""
This file contains functions to be used for competitive programming problems.

For speed purposes, inputs are not checked for validity!

The imports present at the top of functions should be put at the top of your
file, they should not stay in the function itself.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


VertexT = TypeVar("VertexT")


# ############################# GRAPH ALGORITHMS ##############################
# Commonly used variables:
# - adjlist: list[list[int | tuple[int, float]]]
#   Can be used to represent both directed and undirected graphs.
#   Vertices are 0-indexed.
#   For unweighted graphs, adjlist contains lists of integers: to_vertex.
#   For weighted graphs, adjlist contains lists of tuples: (to_vertex, weight).

# ALGORITHMS FOR UNDIRECTED UNWEIGHTED GRAPHS


def reachable_vertices(adjlist: list[list[int]], start: int) -> set[int]:
    """Return the set of vertices that are reachable from start.

    A vertex is reachable if there is a path from start to that vertex.

    This function supports:
    - Undirected unweighted graphs
    - Directed unweighted graphs

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        The set of vertices that are reachable from start.
            Length: Amount of reachable vertices
    """
    reachable = {start}
    unchecked = [start]
    while unchecked:
        u = unchecked.pop()
        for v in adjlist[u]:
            if v not in reachable:
                reachable.add(v)
                unchecked.append(v)
    return reachable


def connected_components(adjlist: list[list[int]]) -> list[set[int]]:
    """Return a list of disjoint sets representing connected components.

    A connected component is a set of vertices such that there is a path
    between every pair of vertices in the set.

    This function supports:
    - Undirected unweighted graphs

    Complexity: O(V + E)

    Notes:
    - For directed graphs, use Tarjan's algorithm.

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)

    Returns:
        A list of disjoint sets representing connected components.
            Length: Amount of connected components
            Length of inner sets: Amount of vertices in the component
    """
    components = []
    unassigned = set(range(len(adjlist)))
    while unassigned:
        component = reachable_vertices(adjlist, unassigned.pop())
        components.append(component)
        unassigned.difference_update(component)
    return components


def bfs(adjlist: list[list[int]], start: int) -> tuple[list[float], list[int]]:
    """Calculate the shortest distance from the start to all other vertices.

    If a vertex is unreachable, its distance will be set to infinity.

    This function supports:
    - Undirected unweighted graphs
    - Directed unweighted graphs

    Complexity: O(V + E)

    Notes:
    - For weighted graphs, use Dijkstra's algorithm or A*.

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
        - List mapping each vertex to the shortest distance from the start.
            If the vertex is unreachable, it will be set to infinity.
            Length: V
        - List mapping each vertex to its parent in the shortest path.
            If the vertex is unreachable, it will be set to -1.
            Length: V
    """
    from collections import deque
    from math import inf, isinf

    dist = [inf] * len(adjlist)
    dist[start] = 0
    parent = [-1] * len(adjlist)
    queue = deque((start,))
    while queue:
        u = queue.popleft()
        for v in adjlist[u]:
            if isinf(dist[v]):  # we didn't visit this vertex yet
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


def bfs_v_type(
    move: Callable[[VertexT], Iterable[VertexT]], start: VertexT
) -> tuple[defaultdict[VertexT, float], defaultdict[VertexT, VertexT | None]]:
    """Calculate the shortest distance from the start to all other vertices.

    If a vertex is unreachable, its distance will be set to infinity.

    This function supports:
    - Undirected unweighted graphs
    - Directed unweighted graphs

    Complexity: O(V + E)

    Notes:
    - For weighted graphs, use Dijkstra's algorithm or A*.
    - This variant of breadth-first search is made for variable vertex types.
        It could be useful in cases where the adjacency list would be very
        large or even infinite if it were built fully. In the latter case, the
        move() function could end the search early to prevent this.

    Args:
        move: A function that takes a vertex v and returns an iterable of
            neighbours.
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
        - Defaultdict mapping each vertex to the shortest distance from the
            start. Warning: If the vertex is unreachable, it will not be
            present in the dict, but when indexing it, the value returned is
            infinity.
        - Defaultdict mapping each vertex to its parent in the shortest path.
            Warning: If the vertex is unreachable, it will not be present in
            the dict, but when indexing it, the value returned is None.
    """
    from collections import defaultdict, deque
    from math import inf

    dist: defaultdict[VertexT, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: defaultdict[VertexT, VertexT | None] = defaultdict(lambda: None)
    queue: deque[VertexT] = deque((start,))
    while queue:
        u = queue.popleft()
        for v in move(u):
            if v not in dist:  # we didn't visit this vertex yet
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


# ALGORITHMS FOR DIRECTED UNWEIGHTED GRAPHS


def reverse_digraph(adjlist: list[list[int]]) -> list[list[int]]:
    """Returns a digraph where the edges are reversed.

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)

    Returns:
        A digraph where the edges are reversed.
            Length: V
            Length of inner lists: indegree(v)
    """
    V = len(adjlist)
    rev_adjlist = [[] for _ in range(V)]
    for u in range(V):
        for v in adjlist[u]:
            rev_adjlist[v].append(u)
    return rev_adjlist


def tarjan(adjlist: list[list[int]]) -> tuple[int, list[int]]:
    """Tarjan's algorithm finds all strongly connected components in a digraph.

    A strongly connected component is a set of vertices such that there is a
    path between every pair of vertices in the set.

    This algorithm assigns component numbers in a such a way that the
    "component graph" is in reverse topological ordering. That is, if there is
    an edge from v to w, then components[v] >= components[w].

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

    Notes:
    - For undirected graphs, use the connected_components() function.

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)

    Returns:
        Tuple containing:
        - The amount of strongly connected components.
        - A list such that for each index v, the corresponding value is the
            component number of v.
            Length: V
    """
    V = len(adjlist)
    low = {}  # lowest reachable depth from a vertex
    stack = []  # stack of candidate vertices for the current component
    call_stack = []  # stack of (vertex, child_idx, depth) tuples
    # child_idx is the index of the next child to visit.
    # depth is the depth at which the vertex was first visited in the DFS.
    amount_components = 0
    components = [-1] * V
    for u in range(V):
        call_stack.append((u, 0, len(low)))
        while call_stack:
            v, child_idx, depth = call_stack.pop()
            if child_idx == 0:  # vertex was just added, did we visit yet?
                if v in low:  # we already visited this vertex
                    continue
                low[v] = depth  # we did not visit yet, continue DFS
                stack.append(v)
            elif child_idx > 0:  # later times checking this vertex
                low[v] = min(low[v], low[adjlist[v][child_idx - 1]])
            if child_idx < len(adjlist[v]):  # still children to visit
                call_stack.append((v, child_idx + 1, depth))
                call_stack.append((adjlist[v][child_idx], 0, len(low)))
                continue
            if depth == low[v]:  # we found a component
                while True:
                    w = stack.pop()
                    components[w] = amount_components
                    low[w] = V
                    if w == v:
                        break
                amount_components += 1
    return amount_components, components


def component_graph(
    adjlist: list[list[int]], amount_components: int, components: list[int]
) -> list[list[int]]:
    """Construct the component digraph of a digraph.

    The strongly connected components of the given digraph are the vertices in
    the returned component digraph.

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

    Notes:
    - Use Tarjan's algorithm to find amount_components and components.

    Args:
        adjlist: The adjacency list of the digraph.
            Length: V
            Length of inner lists: degree(v)
        amount_components: The amount of strongly connected components.
        components: A list such that for each index v, the corresponding value
            is the component number of v.
            Length: V

    Returns:
        The adjacency list of the component digraph.
            Length: amount_components
            Length of inner lists: degree(component)
    """
    component_adjlist = [[] for _ in range(amount_components)]
    for u, nbrs in enumerate(adjlist):
        for v in nbrs:
            if components[u] != components[v]:  # edge between components
                component_adjlist[components[u]].append(components[v])
    for component_u, nbrs in enumerate(component_adjlist):
        component_adjlist[component_u] = list(set(nbrs))  # remove duplicates
    return component_adjlist


def kahn(adjlist: list[list[int]], rev_adjlist: list[list[int]]) -> list[int]:
    """Kahn's algorithm for performing a topological sort of a given tree.

    A topological sorting is an ordering of vertices such that for every edge
    from u to v, u comes before v in the ordering.

    If the given graph is not a directed acyclic graph (DAG), the algorithm
    returns a list with less than V elements. Thus, this is an easy way to
    check if a graph is a DAG.

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

    Notes:
    - Use the reverse_digraph() function to find rev_adjlist.

    Args:
        adjlist: The tree represented as an adjacency list.
            Length: V
            Length of inner lists: degree(v)
        rev_adjlist: The adjacency list of the reversed graph.
            Length: V
            Length of inner lists: indegree(v)

    Returns:
        A list of vertices, sorted in topological order.
            Length: Amount of vertices in the topological order
    """
    num_parents = [len(parents) for parents in rev_adjlist]
    stack = [v for v in range(len(adjlist)) if num_parents[v] == 0]
    toposort = []
    while stack:
        v = stack.pop()
        toposort.append(v)
        for child in adjlist[v]:
            num_parents[child] -= 1
            if num_parents[child] == 0:
                stack.append(child)
    return toposort


# ALGORITHMS FOR UNDIRECTED WEIGHTED GRAPHS


def dijkstra(
    adjlist: list[list[tuple[int, float]]], start: int
) -> tuple[list[float], list[int]]:
    """Dijkstra's algorithm finds the shortest path from start to all vertices.

    If a vertex is unreachable, its distance will be set to infinity, and its
    parent will be set to -1.

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    Complexity: O((V + E) * log(V))

    Notes:
    - For unweighted graphs, use the bfs() function.

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
        - List mapping each vertex to the shortest distance from the start.
            If the vertex is unreachable, it will be set to infinity.
            Length: V
        - List mapping each vertex to its parent in the shortest path.
            If the vertex is unreachable, it will be set to -1.
            Length: V
    """
    from heapq import heappop, heappush
    from math import inf

    dist = [inf] * len(adjlist)
    dist[start] = 0
    parent = [-1] * len(adjlist)
    # Store the distance to each vertex to ensure the closest is popped first.
    prioq = []  # (dist, vertex)
    heappush(prioq, (0, start))
    while prioq:
        dist_u, u = heappop(prioq)
        if dist[u] < dist_u:  # we already visited this vertex
            continue
        for v, w in adjlist[u]:
            dist_v = dist_u + w
            if dist_v < dist[v]:
                dist[v] = dist_v
                parent[v] = u
                heappush(prioq, (dist_v, v))
    return dist, parent


def dijkstra_v_type(
    move: Callable[[VertexT], Iterable[tuple[VertexT, float]]], start: VertexT
) -> tuple[defaultdict[VertexT, float], defaultdict[VertexT, VertexT | None]]:
    """Dijkstra's algorithm finds the shortest path from start to all vertices.

    If a vertex is unreachable, its distance will be set to infinity, and its
    parent will be set to -1.

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    Complexity: O((V + E) * log(V))

    Notes:
    - For unweighted graphs, use the bfs() function.
    - This variant of Dijkstra's algorithm is made for variable vertex types.
        It could be useful in cases where the adjacency list would be very
        large or even infinite if it were built fully. In the latter case, the
        move() function could end the search early to prevent this.

    Args:
        move: A function that takes a vertex v and returns an iterable of
            (neighbour, weight) tuples.
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
        - Defaultdict mapping each vertex to the shortest distance from the
            start. Warning: If the vertex is unreachable, it will not be
            present in the dict, but when indexing it, the value returned is
            infinity.
        - Defaultdict mapping each vertex to its parent in the shortest path.
            Warning: If the vertex is unreachable, it will not be present in
            the dict, but when indexing it, the value returned is None.
    """
    from collections import defaultdict
    from heapq import heappop, heappush
    from math import inf

    dist: defaultdict[VertexT, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: defaultdict[VertexT, VertexT | None] = defaultdict(lambda: None)
    # Store the distance to each vertex to ensure the closest is popped first.
    prioq: list[tuple[float, VertexT]] = []  # (dist, vertex)
    prioq = []  # (dist, vertex)
    heappush(prioq, (0, start))
    while prioq:
        dist_u, u = heappop(prioq)
        if dist[u] < dist_u:  # we already visited this vertex
            continue
        for v, w in move(u):
            dist_v = dist_u + w
            if v not in dist or dist_v < dist[v]:
                dist[v] = dist_v
                parent[v] = u
                heappush(prioq, (dist_v, v))
    return dist, parent


def a_star(
    adjlist: list[list[tuple[int, float]]],
    start: int,
    goal: int,
    heuristic: Callable[[int], float],
) -> tuple[float, list[int]]:
    """The A* algorithm finds the shortest path from a start to a goal vertex.

    Warning: This variant is usually slower than the A* algorithm that works
    with variable vertex types, since the search is usually ended early. In
    these cases, is more memory efficient to specify a move function than
    spacify the whole adjlist, since you don't have to build the whole
    adjacency list in advance.

    A* is a generalization of Dijkstra's algorithm that uses a heuristic to
    speed up the search. The heuristic must be admissable for the algorithm to
    find the optimal path.

    An admissable heuristic is a function that returns an estimate to the goal
    vertex that is always <= to the actual distance. For example, the Manhattan
    distance is an admissable heuristic for the shortest path problem on a
    grid graph.

    For a slow heuristic, implementing a cache is recommended.

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    Complexity: O((V + E) * log(V))

    Notes:
    - For unweighted graphs, use the bfs() function.

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.
        goal: The vertex to find a path to.
        heuristic: A function that takes a vertex v and returns an estimate to
            the goal vertex.

    Returns:
        Tuple containing:
        - The shortest distance from the start to the goal vertex. If the goal
            vertex is unreachable, it will be set to infinity.
        - The shortest path from the start to the goal, represented as a list
            of vertices along which to walk. If the goal vertex is ureachable,
            the list will equal [goal].
            Length: Amount of vertices in the path
    """
    from heapq import heappop, heappush
    from math import inf

    dist = [inf] * len(adjlist)
    dist[start] = 0
    parent = [-1] * len(adjlist)
    # Store the distance + estimate to each vertex to ensure the one expected
    # to be closest is popped first.
    prioq = []  # (dist + est, dist, vertex)
    heappush(prioq, (heuristic(start), 0, start))
    while prioq:
        _, dist_u, u = heappop(prioq)
        if u == goal:
            break
        if dist[u] < dist_u:  # we already visited this vertex
            continue
        for v, w in adjlist[u]:
            dist_v = dist_u + w
            if dist_v < dist[v]:
                dist[v] = dist_v
                parent[v] = u
                heappush(prioq, (dist_v + heuristic(v), dist_v, v))
    path = [goal]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    path.reverse()
    return dist[goal], path


def a_star_v_type(
    move: Callable[[VertexT], Iterable[tuple[VertexT, float]]],
    start: VertexT,
    goal: VertexT,
    heuristic: Callable[[VertexT], float],
) -> tuple[float, list[VertexT]]:
    """The A* algorithm finds the shortest path from a start to a goal vertex.

    A* is a generalization of Dijkstra's algorithm that uses a heuristic to
    speed up the search. The heuristic must be admissable for the algorithm to
    find the optimal path.

    An admissable heuristic is a function that returns an estimate to the goal
    vertex that is always <= to the actual distance. For example, the Manhattan
    distance is an admissable heuristic for the shortest path problem on a
    grid graph.

    For a slow heuristic, implementing a cache is recommended.

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    Complexity: O((V + E) * log(V))

    Notes:
    - For unweighted graphs, use the bfs() function.

    Args:
        move: A function that takes a vertex v and returns an iterable of
            (neighbour, weight) tuples.
        start: The vertex to start the search from.
        goal: The vertex to find a path to.
        heuristic: A function that takes a vertex v and returns an estimate to
            the goal vertex.

    Returns:
        Tuple containing:
        - The shortest distance from the start to the goal vertex. If the goal
            vertex is unreachable, it will be set to infinity.
        - The shortest path from the start to the goal, represented as a list
            of vertices along which to walk. If the goal vertex is ureachable,
            the list will equal [goal].
            Length: Amount of vertices in the path
    """
    from collections import defaultdict
    from heapq import heappop, heappush
    from math import inf

    dist: dict[VertexT, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: dict[VertexT, VertexT | None] = defaultdict(lambda: None)
    # Store the distance + estimate to each vertex to ensure the one expected
    # to be closest is popped first. Stores pairs of (dist + est, dist, vertex)
    prioq: list[tuple[float, float, VertexT]] = []
    heappush(prioq, (heuristic(start), 0, start))
    while prioq:
        _, dist_u, u = heappop(prioq)
        if u == goal:
            break
        if dist[u] < dist_u:  # we already visited this vertex
            continue
        for v, w in move(u):
            dist_v = dist_u + w
            if v not in dist or dist_v < dist[v]:
                dist[v] = dist_v
                parent[v] = u
                heappush(prioq, (dist_v + heuristic(v), dist_v, v))
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore
    path.reverse()
    return dist[goal], path


def prim(
    adjlist: list[list[tuple[int, float]]], start: int = 0
) -> list[list[tuple[int, float]]]:
    """Prim's algorithm finds a set of edges that form a minimal spanning tree.

    A minimal spanning tree (MST) is a subset of the edges of a weighted graph
    that connects all vertices together, without any cycles, and with the
    smallest possible total edge weight.

    If the graph is not connected, the algorithm returns an MST that connects
    to the start vertex. If the graph is connected, supplying a start vertex
    will not change the result or influence performance.

    This function supports:
    - Undirected weighted graphs

    Complexity: O((V + E) * log(V))

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from. Do not supply this
            argument if the graph is connected. Defaults to 0.

    Returns:
        The minimal spanning tree of the graph as an adjacency list.
            Length: V
            Length of inner lists: Amount of edges in the MST that have v as
                an endpoint
    """
    from heapq import heappop, heappush
    from math import inf

    V = len(adjlist)
    # Keep track of the vertices currently connected to the MST.
    visited = [False] * V
    visited[start] = True
    # Store the distance to each vertex to speed up choosing the right edges.
    dist = [inf] * V
    dist[start] = 0
    # Use a priority queue of tuples with (weight, from_vertex, to_vertex).
    prioq = []
    for v, w in adjlist[start]:
        dist[v] = w
        heappush(prioq, (w, start, v))
    # Repeatedly add the smallest edge connected to an unvisited vertex.
    mst = [[] for _ in range(V)]
    for _ in range(V - 1):
        if not prioq:
            break  # graph is not connected
        w, u, v = heappop(prioq)
        while visited[v]:
            w, u, v = heappop(prioq)
        mst[u].append((v, w))
        mst[v].append((u, w))
        visited[v] = True
        for u, w in adjlist[v]:
            # Filtering edges based on distance prematurely is not necessary,
            # but speeds up the algorithm since we don't have to push as many
            # edges to the priority queue (I tested this thoroughly).
            if not visited[u] and w < dist[u]:
                dist[u] = w
                heappush(prioq, (w, v, u))
    return mst


class UnionFind:
    """A union-find data structure divides a graph's vertices into clusters."""

    def __init__(self, V: int) -> None:
        """Initialize the union-find data structure.

        Args:
            V: The amount of vertices in the graph.
        """
        self.parents = list(range(V))
        self.sizes = [1] * V

    def find(self, v: int) -> int:
        """Find the root of the disjoint set that the given vertex is in.

        This function uses internal caching to speed up future find operations.

        Args:
            v: The vertex to find the root of.

        Returns:
            The root of the disjoint set that the given vertex is in.
        """
        if self.parents[v] == v:
            return v
        self.parents[v] = self.find(self.parents[v])
        return self.parents[v]

    def union(self, u: int, v: int) -> bool:
        """Unite the sets that u and v are in if they are disjoint.

        Args:
            u: The first vertex to unite.
            v: The second vertex to unite.

        Returns:
            Whether u and v are in disjoint sets.
        """
        i_root, j_root = self.find(u), self.find(v)
        if i_root == j_root:
            return False
        if self.sizes[i_root] < self.sizes[j_root]:
            i_root, j_root = j_root, i_root
        # j_root is now the smaller tree, so hang it under i_root.
        self.parents[j_root] = i_root
        self.sizes[i_root] += self.sizes[j_root]
        return True


def kruskal(
    V: int, edge_list: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """Kruskal's algorithm finds a minimal spanning tree of a graph.

    Warning: Prim's algorithm is shorter and often faster in practice.

    A minimal spanning tree (MST) is a subset of the edges of a weighted graph
    that connects all vertices together, without any cycles, and with the
    smallest possible total edge weight.

    If the graph is not connected, the algorithm returns TODO.
    If the graph is connected, TODO.

    This function supports:
    - Undirected weighted graphs

    Complexity: TODO

    Args:
        V: The amount of vertices in the graph.
        edge_list: An list of undirected edges as tuples containing:
            - Vertex 1 of the edge.
            - Vertex 2 of the edge.
            - Weight of the edge.

    Returns:
        The minimal spanning tree of the graph as a list of edges.
    """
    from operator import itemgetter

    edge_list.sort(key=itemgetter(2))  # sort edges by weight
    UF = UnionFind(V)
    # Continuously add edges that don't result in a cycle.
    return [(u, v, w) for u, v, w in edge_list if UF.union(u, v)]


# ALGORITHMS FOR DIRECTED WEIGHTED GRAPHS

# =============================================================================
# UPDATED COMMENTS AND IMPLEMENTED UNIT TESTS UP TO THIS POINT OF THE
# CHEATSHEET. EVERYTHING BELOW THIS POINT IS UNTESTED.
# =============================================================================


# TODO This algorithm is personalized, but unit tests haven't been written yet.
def edmonds_karp(
    adjlist: list[list[tuple[int, float]]], s: int, t: int
) -> tuple[float, list[list[tuple[int, float]]]]:
    """Edmonds-Karp's algorithm finds the maximum flow from s to t.

    Warning: if there is an edge from u to v and from v to u, then the flow
    along one of these edges in the returned adjacency list will be negative
    (or both will be zero).

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    Complexity: O(V * E^2)

    Args:
        adjlist: The adjacency list of the graph. The weights represent the
            capacities of the edges.
            Length: V
            Length of inner lists: degree(v)
        s: The source vertex.
        t: The sink vertex.

    Returns:
        Tuple containing:
        - The maximum flow from s to t.
        - An adjacency list representing the flow along each edge, which has
            the same structure as the input adjacency list.
            Length: V
            Length of inner lists: degree(v)
    """
    from collections import deque
    from math import inf

    if s == t:
        return inf, [[(v, 0) for v, _ in nbrs] for nbrs in adjlist]
    V = len(adjlist)
    cap: list[list[float]] = [[0] * V for _ in range(V)]  # remaining capacity
    for u, nbrs in enumerate(adjlist):
        for v, w in nbrs:
            cap[u][v] = w
    adjlist_un = [[] for _ in range(V)]  # undirected and unweighted
    for u, nbrs in enumerate(adjlist):
        for v, w in nbrs:
            adjlist_un[u].append(v)
            adjlist_un[v].append(u)

    def bfs(
        adjlist_un: list[list[int]], cap: list[list[float]], s: int, t: int
    ) -> tuple[float, list[int]]:
        parent = [-1] * V
        queue = deque(((s, inf),))
        while queue:
            u, curr_flow = queue.popleft()
            for v in adjlist_un[u]:
                if (
                    parent[v] == -1  # we didn't visit this vertex yet
                    and cap[u][v] > 0  # there is still capacity
                ):
                    parent[v] = u
                    new_flow = min(curr_flow, cap[u][v])
                    if v == t:
                        return new_flow, parent
                    queue.append((v, new_flow))
        return 0, parent

    max_flow = 0
    while True:
        # Find a shortest augmenting path.
        new_flow, parent = bfs(adjlist_un, cap, s, t)
        # If no augmenting path was found, we cannot add any more flow.
        if parent[t] == -1:
            break
        # Update the remaining capacity along the augmenting path.
        curr = t
        while curr != s:
            prev = parent[curr]
            cap[prev][curr] -= new_flow
            cap[curr][prev] += new_flow
            curr = prev
        # Update the maximum flow.
        max_flow += new_flow
    # Construct the flow along each edge.
    flow = [[] for _ in range(V)]
    for u, nbrs in enumerate(adjlist):
        for v, w in nbrs:
            flow[u].append((v, w - cap[u][v]))
    return max_flow, flow


def hopcroft_karp(
    adjlist: list[list[int]], left: list[bool]
) -> tuple[int, list[int]]:
    """Hopcroft-Karp's algorithm finds a maximal matching in a bipartite graph.

    Warning: This algorithm can be replaced by Edmonds-Karp in most cases.

    This function supports:
    - Unweighted bipartite graphs

    Complexity: TODO

    Args:
        adjlist: The adjacency list of the bipartite graph.
            Length: V
            Length of inner lists: degree(v)
        left: A boolean array which gives the left part of the bipartite graph.
            If a vertex w is in adjlist[v], then precisely one of left[v] or
            left[w] should be True.
            Length: V

    Returns:
        Tuple containing:
        - The amount of edges in the maximal matching.
        - A list containing the matched vertex for every vertex in the graph,
            where a -1 indicates an unmatched vertex.
            Length: V
    """
    from collections import deque

    V = len(adjlist)
    matching = [-1] * V
    while True:
        # If our matching changes, we need another iteration.
        extended = False
        # Do a BFS starting at all unmatched left nodes.
        sources = [v for v in range(V) if matching[v] == -1 and left[v]]
        # Keep track of your immediate predecessor in the BFS.
        # -1 means an unvisited node. Sources have themselves as predecessor.
        bfs_pred = [-1] * V
        # Keep track of which BFS source started your subtree.
        # -1 means an unvisited node.
        bfs_source = [-1] * V
        for v in sources:
            bfs_pred[v] = v
            bfs_source[v] = v
        queue = deque(sources)
        while queue:
            curr = queue.popleft()
            # If the node that started this subtree has been matched, stop
            # growing the BFS.
            if matching[bfs_source[curr]] != -1:
                continue
            for nbr in adjlist[curr]:
                if matching[nbr] == -1:
                    # We have found an augmenting path.
                    # Use a clever loop to use the path to update the matching.
                    while nbr != -1:
                        matching[nbr], matching[curr], nbr, curr = (
                            curr,
                            nbr,
                            matching[curr],
                            bfs_pred[curr],
                        )
                    extended = True
                    break
                else:
                    new = matching[nbr]
                    if bfs_pred[new] == -1:
                        bfs_pred[new] = curr
                        bfs_source[new] = bfs_source[curr]
                        queue.append(new)
        if not extended:
            break
    return sum(x != -1 for x in matching) // 2, matching


# ALGORITHMS FOR TREES
# Note: The trees represented as adjacency lists in the following algorithms
# are formally called directed trees, which are directed acyclic graphs (DAGs)
# whose underlying undirected graph is a tree.


def depthfirst_pre_order(adjlist: list[list[int]], root: int) -> Iterator[int]:
    """Yield vertices from the tree rooted at root in pre-order.

    A pre-order traversal visits the root before its children.

    This function supports:
    - Directed trees

    Complexity: O(V)

    Args:
        adjlist: The adjacency list of the tree.
            Length: V
            Length of inner lists: degree(v)
        root: Vertex at which to start the traversal.

    Yields:
        Vertices in the tree rooted at root in pre-order.
    """
    stack = [root]
    while stack:
        v = stack.pop()
        yield v
        stack.extend(reversed(adjlist[v]))


def depthfirst_post_order(
    adjlist: list[list[int]], root: int
) -> Iterator[int]:
    """Yield vertices from the tree rooted at root in post-order.

    A post-order traversal visits the root after its children.

    This function supports:
    - Directed trees

    Complexity: O(V)

    Args:
        adjlist: The adjacency list of the tree.
            Length: V
            Length of inner lists: degree(v)
        root: Vertex at which to start the traversal.

    Yields:
        Vertices in the tree rooted at root in post-order.
    """
    stack = [root]
    visited = []
    while stack:
        v = stack.pop()
        visited.append(v)
        stack.extend(adjlist[v])
    return reversed(visited)


# ################################## SEARCH ###################################
# Note: To search for the index of a specific value in a sorted list, always
# use the bisect module from the standard library. It is faster and more
# concise than a binary search implementation in Python code.


def binary_search_discrete_func_left(
    f: Callable[[int], float], output: float, lower: int, upper: int
) -> int:
    """Calculate the smallest input that maps to the target output.

    If multiple inputs map to the target, the leftmost one will be returned.
    If no input maps exactly to the target, the returned input x* will be
        rounded to the right (!) and satisfy:
        f(x* - 1) < output < f(x*)

    Making your function f cache its results is recommended, as this algorithm
    may call it twice times for the same input (this will happen at most once).

    Args:
        f: A discrete function that is monotonically increasing.
        output: The target output.
        lower: An inclusive lower bound for the target input.
        upper: An inclusive upper bound for the target input.

    Returns:
        The smallest input satisfying the conditions above.
    """
    while upper - lower > 1:
        guess = (lower + upper) // 2
        if f(guess) < output:
            lower = guess
        else:
            upper = guess
    if f(lower) >= output:
        return lower
    return upper


def binary_search_discrete_func_right(
    f: Callable[[int], float], output: float, lower: int, upper: int
) -> int:
    """Calculate the largest input that maps to the target output.

    If multiple inputs map to the target, the rightmost one will be returned.
    If no input maps exactly to the target, the returned input x* will be
        rounded to the left (!) and satisfy:
        f(x*) < output < f(x* + 1)

    Making your function f cache its results is recommended, as this algorithm
    may call it twice times for the same input (this will happen at most once).

    Args:
        f: A discrete function that is monotonically increasing.
        output: The target output.
        lower: An inclusive lower bound for the target input.
        upper: An inclusive upper bound for the target input.

    Returns:
        The largest input satisfying the conditions above.
    """
    while upper - lower > 1:
        guess = (lower + upper) // 2
        if f(guess) <= output:
            lower = guess
        else:
            upper = guess
    if f(upper) <= output:
        return upper
    return lower


def binary_search_continuous_func_left(
    f: Callable[[float], float],
    output: float,
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """Calculate the smallest input that maps to the target output.

    If multiple inputs map to the target, the leftmost one will be returned.

    The returned input x* satisfies:
        x* - f_inv(output) <= abs_tol or (x* - f_inv(output)) / x* <= rel_tol

    Args:
        f: A continuous function that is monotonically increasing.
        output: The target output.
        lower: A lower bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        upper: An upper bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        abs_tol: The absolute tolerance for the target input.
        rel_tol: The relative tolerance for the target input.

    Returns:
        The smallest input satisfying the conditions above.
    """
    while (
        upper - lower >= abs_tol and abs(upper - lower) >= abs(upper) * rel_tol
    ):
        m = (lower + upper) / 2
        if f(m) < output:
            lower = m
        else:
            upper = m
    return lower


def binary_search_continuous_func_right(
    f: Callable[[float], float],
    output: float,
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """Calculate the largest input that maps to the target output.

    If multiple inputs map to the target, the rightmost one will be returned.

    The returned input x* satisfies:
        f_inv(output) - x* <= abs_tol or (f_inv(output) - x*) / x* <= rel_tol

    Args:
        f: A continuous function that is monotonically increasing.
        output: The target output.
        lower: A lower bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        upper: An upper bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        abs_tol: The absolute tolerance for the target input.
        rel_tol: The relative tolerance for the target input.

    Returns:
        The largest input satisfying the conditions above.
    """
    while (
        upper - lower >= abs_tol and abs(upper - lower) >= abs(upper) * rel_tol
    ):
        m = (lower + upper) / 2
        if f(m) <= output:
            lower = m
        else:
            upper = m
    return upper


def ternary_search_discrete_func_left(
    f: Callable[[int], float], lower: int, upper: int
) -> int:
    """Calculate the smallest input minimizing a discrete unimodal function.

    If multiple minima exist, the leftmost one will be returned.

    Making your function f cache its results is highly recommended, as this
        algorithm may call it multiple times for the same input.

    Args:
        f: A discrete function that is strictly decreasing followed by
            monotonically increasing. In other words, the left side of the
            function can't have any plateaus!
        lower: An inclusive lower bound for the target input.
        upper: An inclusive upper bound for the target input.

    Returns:
        The smallest input satisfying the conditions above.
    """
    while upper - lower > 2:
        m1 = (2 * lower + upper) // 3
        m2 = (lower + 2 * upper) // 3
        if f(m1) > f(m2):
            lower = m1
        else:
            upper = m2
    if f(lower) <= f(lower + 1):
        return lower
    if f(upper - 1) <= f(upper):
        return upper - 1
    return upper


def ternary_search_discrete_func_right(
    f: Callable[[int], float], lower: int, upper: int
) -> int:
    """Calculate the largest input minimizing a discrete unimodal function.

    If multiple minima exist, the rightmost one will be returned.

    Making your function f cache its results is highly recommended, as this
        algorithm may call it multiple times for the same input.

    Args:
        f: A discrete function that is monotonically decreasing followed by
            strictly increasing. In other words, the right side of the function
            can't have any plateaus!
        lower: An inclusive lower bound for the target input.
        upper: An inclusive upper bound for the target input.

    Returns:
        The largest input satisfying the conditions above.
    """
    while upper - lower > 2:
        m1 = (2 * lower + upper) // 3
        m2 = (lower + 2 * upper) // 3
        if f(m1) >= f(m2):
            lower = m1
        else:
            upper = m2
    if f(upper - 1) >= f(upper):
        return upper
    if f(lower) >= f(lower + 1):
        return lower + 1
    return lower


def ternary_search_continuous_func_left(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """Calculate the smallest input minimizing a continuous unimodal function.

    If multiple minima exist, the leftmost one will be returned.

    The returned input x* satisfies:
        x* - argmin(f) <= abs_tol or (x* - argmin(f)) / x* <= rel_tol

    Args:
        f: A continuous function that is strictly decreasing followed by
            monotonically increasing. In other words, the left side of the
            function can't have any plateaus!
        lower: A lower bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        upper: An upper bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        abs_tol: The absolute tolerance for the returned input.
        rel_tol: The relative tolerance for the returned input.

    Returns:
        The smallest input satisfying the conditions above.
    """
    while (
        upper - lower >= abs_tol and abs(upper - lower) >= abs(upper) * rel_tol
    ):
        m1 = (2 * lower + upper) / 3
        m2 = (lower + 2 * upper) / 3
        if f(m1) > f(m2):
            lower = m1
        else:
            upper = m2
    return lower


def ternary_search_continuous_func_right(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """Calculate the largest input minimizing a continuous unimodal function.

    If multiple minima exist, the rightmost one will be returned.

    The returned input x* satisfies:
        argmin(f) - x* <= abs_tol or (argmin(f) - x*) / x* <= rel_tol

    Args:
        f: A continuous function that is monotonically decreasing followed by
            strictly increasing. In other words, the right side of the function
            can't have any plateaus!
        lower: A lower bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        upper: An upper bound for the target input. Whether it is inclusive or
            exclusive doesn't matter.
        abs_tol: The absolute tolerance for the returned input.
        rel_tol: The relative tolerance for the returned input.

    Returns:
        The largest input satisfying the conditions above.
    """
    while (
        upper - lower >= abs_tol and abs(upper - lower) >= abs(upper) * rel_tol
    ):
        m1 = (2 * lower + upper) / 3
        m2 = (lower + 2 * upper) / 3
        if f(m1) >= f(m2):
            lower = m1
        else:
            upper = m2
    return upper


# ################################### MISC ####################################


class SegTree:
    """A segment tree speeds up associative operations of consecutive elements.

    More specifically, it is a binary tree that makes associative operations
    between multiple consecutive elements in a sequence faster. The bottom
    layer is the original list, and any node above the bottom layer contains
    the sum/minimum/maximum/etc. of the two nodes beneath.

    This implementation is for range sum queries and point updates.
    """

    def __init__(
        self, lst: list[float] | None = None, length: int | None = None
    ) -> None:
        """Initialize a segment tree.

        Either lst or length must be given. If lst is given, length is ignored,
        and we build a segment tree with underlying list lst. If no list is
        given, length must be given, and we build a segment tree with an
        underlying all-zeros list of that length.

        Args:
            lst: The list to build the segment tree from.
            length: The length of the list to build the segment tree from.
        """
        if lst is not None:
            self.len = len(lst)
            self.t = [0.0] * (2 * self.len)
            self.t[self.len :] = lst
            for i in reversed(range(1, self.len)):
                self.t[i] = self.t[i * 2] + self.t[i * 2 + 1]
        elif length is not None:
            self.len = length
            self.t = [0.0] * (2 * self.len)

    def modify(self, idx: int, val: float) -> None:
        """Set the value at the given index to val.

        Args:
            idx: The index to set the value of.
            val: The value to change the index to.
        """
        idx += self.len
        self.t[idx] = val
        while idx > 1:
            self.t[idx // 2] = self.t[idx] + self.t[idx ^ 1]
            idx //= 2

    def query(self, left: int, right: int) -> float:
        """Get the sum of all values in the range [left, right).

        Args:
            left: The left index of the range.
            right: The right index of the range.

        Returns:
            The sum of all values in the range [left, right).
        """
        res = 0
        left += self.len
        right += self.len
        while left < right:
            if left & 1:
                res += self.t[left]
                left += 1
            if right & 1:
                right -= 1
                res += self.t[right]
            left //= 2
            right //= 2
        return res


def aho_corasick(
    words: Iterable[str], target: str
) -> Iterator[tuple[str, int]]:
    """Aho-Corasick's algorithm finds all matches of all words in a string.

    Complexity: O(n + m + z)
        n: The amount of characters in the target string.
        m: The sum of the amount of characters in the words to search.
        z: The amount of total macthes returned.
        Note that since we have to go through all characters and matches at
        least once, this is the theoretical limit in terms of efficiency.

    Args:
        words: The words to search for in the target string.
            Length: K
            Length of inner strings: m_k
        target: The string to search for the words in.
            Length: n

    Yields:
        Tuple containing:
        - The word found in the target string.
        - The index in the target string of the last letter in word. The
            matches are always returned in order of increasing index_in_target.
    """
    from collections import deque

    # First build the trie: O(m).
    # Each node is represented by a list containing:
    # [0] A dict pointing to the node's children.
    # [1] The suffix link of this node.
    # [2] The output link of this node (or None).
    # [3] A word if the node corresponds to the end of a word (or None).
    trie = [{}, None, None, None]
    for word in words:
        # Traverse the trie, going into branch c at every step.
        curr_node = trie
        for c in word:
            curr_node = curr_node[0].setdefault(c, [{}, None, None, None])
        # Save the word at the ending node.
        curr_node[3] = word

    # Now add the suffix links and output links to the trie: O(n).
    # Perform a BFS of the trie, excluding the root:
    # - Call the character leading to the node c.
    # - Call the node where the parent's suffix link points to x.
    # Now we can find the suffix link of the node.
    # - If the node xc exists, the node's suffix link points to xc.
    # - Otherwise, set x to the node pointed at by x's suffix link and repeat.
    # Now we can find the output link of the node.
    # - Call the node where node's suffix link points to y.
    # - If y is a word, the node's output link points to y.
    # - Otherwise, the node's output link points to y's output link.
    queue = deque((trie,))
    while queue:
        parent = queue.popleft()
        for c, node in parent[0].items():
            queue.append(node)
            # Add the suffix link.
            if parent is trie:
                node[1] = trie
            else:
                x = parent[1]
                while True:
                    if c in x[0]:
                        node[1] = x[0][c]
                        break
                    if x is trie:
                        node[1] = trie
                        break
                    x = x[1]
            # Add the output link.
            y = node[1]
            node[2] = y if y[3] is not None else y[2]

    # Finally, perform the matching algorithm: O(n + z).
    node = trie
    for i, c in enumerate(target):
        # Follow suffix links.
        while c not in node[0]:
            if node is trie:
                break
            node = node[1]
        node = node[0].get(c, trie)
        # Output a word if this node corresponds to (the end of) a word.
        if node[3] is not None:
            yield (node[3], i)
        # Output all words in the chain of output links starting at this node.
        output_link = node[2]
        while output_link is not None:
            yield (output_link[3], i)
            output_link = output_link[2]
