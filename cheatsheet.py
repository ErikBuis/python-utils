"""
This file contains functions to be used for competitive programming problems.

For speed purposes, inputs are not checked for validity!

The imports present at the top of functions should be put at the top of your
file, they should not stay in the function itself.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


v_type = TypeVar("v_type")


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

    For weighted graphs, implement your own variant of this function.

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        The set of vertices that are reachable from start.
            Length: amount of reachable vertices
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

    For directed graphs, use Tarjan's algorithm.
    For weighted graphs, implement your own variant of reachable_vertices.

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)

    Returns:
        A list of disjoint sets representing connected components.
            Length: amount of connected components
            Length of inner sets: amount of vertices in the component
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

    For weighted graphs, use Dijkstra's algorithm.

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
            List mapping each vertex to the shortest distance from the start.
                If the vertex is unreachable, it will be set to infinity.
            List mapping each vertex to its parent in the shortest path.
                If the vertex is unreachable, it will be set to -1.
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


def kahn(adjlist: list[list[int]], rev_adjlist: list[list[int]]) -> list[int]:
    """Kahn's algorithm for performing a topological sort of a given DAG.

    A topological sorting is an ordering of vertices such that for every edge
    from u to v, u comes before v in the ordering.

    If the given graph is not a DAG, the algorithm returns a list with less
    than V elements. Thus, this is an easy way to check if a graph is a DAG.

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        rev_adjlist: The adjacency list of the reversed graph.
            Length: V
            Length of inner lists: indegree(v)

    Returns:
        A list of vertices, sorted in topological order.
            Length: amount of vertices in the topological order
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


def tarjan(adjlist: list[list[int]]) -> tuple[int, list[int]]:
    """Tarjan's algorithm finds all strongly connected components in a digraph.

    A strongly connected component is a set of vertices such that there is a
    path between every pair of vertices in the set.

    This algorithm assigns component numbers in a such a way that the
    "component graph" is in reverse topological ordering. That is, if there is
    an edge from v to w, then components[v] >= components[w].

    This function supports:
    - Directed unweighted graphs

    For undirected graphs, use the connected components algorithm.

    Complexity: O(V + E)

    Args:
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)

    Returns:
        Tuple containing:
            The amount of strongly connected components.
            A list such that for each index v, the corresponding value is
                the component number of v.
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

    The strongly connected components of the given generated digraph are the
    vertices in the component digraph.

    Note: You should use Tarjan's algorithm to find amount_components and
    components.

    This function supports:
    - Directed unweighted graphs

    Complexity: O(V + E)

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

    For unweighted graphs, use BFS instead.

    Complexity: O((V + E) * log(V))

    Args:
        Adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
            List mapping each vertex to the shortest distance from the start.
                If the vertex is unreachable, it will be set to infinity.
            List mapping each vertex to its parent in the shortest path.
                If the vertex is unreachable, it will be set to -1.
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


def a_star_v_type(
    move: Callable[[v_type], Iterable[tuple[v_type, float]]],
    start: v_type,
    goal: v_type,
    heuristic: Callable[[v_type], float],
) -> tuple[float, list[v_type]]:
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

    For unweighted graphs, use BFS instead.

    Complexity: O((V + E) * log(V))

    Args:
        move: A function that takes a vertex v and returns an iterable of
            (neighbour, weight) tuples.
        start: The vertex to start the search from.
        goal: The vertex to find a path to.
        heuristic: A function that takes a vertex v and returns an estimate to
            the goal vertex.

    Returns:
        Tuple containing:
            The shortest distance from the start to the goal vertex.
                If the goal vertex is unreachable, it will be set to infinity.
            The shortest path from the start to the goal, represented as a
                list of vertices along which to walk. If the goal vertex is
                unreachable, the list will equal [goal].
                Length: amount of vertices in the path
    """
    from collections import defaultdict
    from heapq import heappop, heappush
    from math import inf

    dist: dict[v_type, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: dict[v_type, v_type | None] = defaultdict(lambda: None)
    # Store the distance + estimate to each vertex to ensure the one expected
    # to be closest is popped first. Stores pairs of (dist + est, dist, vertex)
    prioq: list[tuple[float, float, v_type]] = []
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
        mst: The minimal spanning tree of the graph as an adjacency list.
            Length: V
            Length of inner lists: amount of edges in the mst that have v as
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
            The maximum flow from s to t.
            An adjacency list representing the flow along each edge, which has
                the same structure as the input adjacency list.
                Warning: if there is an edge from u to v and from v to u, then
                the flow along one of these edges will be negative (or both
                will be zero).
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


# ############################# TREE ALGORITHMS ###############################
# Commonly used variables:
# - tree: list of lists representing the children of the corresponding vertex.


def depthfirst_pre_order(tree: list[list[int]], root: int) -> Iterator[int]:
    """Yield the vertices from `tree` rooted at `root` in pre-order."""
    stack = [root]
    while stack:
        v = stack.pop()
        yield v
        stack.extend(reversed(tree[v]))


def depthfirst_post_order(tree: list[list[int]], root: int) -> Iterator[int]:
    """Yield the vertices from `tree` rooted at `root` in post-order."""
    stack = [root]
    visited = []
    while stack:
        v = stack.pop()
        visited.append(v)
        stack.extend(tree[v])
    return reversed(visited)


# ################################## SEARCH ###################################
# Note: To search for the index of a specific value in a sorted list, always
# use the bisect module from the standard library. It is faster and more
# concise.


def binary_search_discrete_func(
    f: Callable[[int], float], output: float, lower: int, upper: int
) -> int:
    """
    Calculates the smallest input satisfying `f(input) >= output`.
    `f` must be a monotonically increasing function with a discrete domain.
    `lower` must be an exclusive lower bound for the target input.
    `upper` must be an inclusive upper bound for the target input.
    """
    while upper - lower > 1:
        guess = (lower + upper + 1) // 2
        if f(guess) >= output:
            upper = guess
        else:
            lower = guess
    return upper


def binary_search_continuous_func(
    f: Callable[[float], float],
    output: float,
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """
    Calculates an input satisfying:
        `f(input - abs_tol) <= output <= f(input + abs_tol)` or
        `f(input - input*rel_tol) <= output <= f(input + input*rel_tol)`
    `f` must be a monotonically increasing function with a continuous domain.
    `lower` and `upper` must be lower and upper bounds for the target input.
        Whether they are inclusive or exclusive doesn't matter.
    """
    while upper - lower >= abs_tol and abs((upper - lower) / upper) >= rel_tol:
        guess = (lower + upper) / 2
        if f(guess) >= output:
            upper = guess
        else:
            lower = guess
    return upper


def ternary_search_discrete_func(
    f: Callable[[int], float], lower: int, upper: int
) -> int:
    """
    Calculates the smallest input satisfying:
        `f(input - 1) >= min(f) <= f(input + 1)`
    `f` must be a unimodal (strictly decreasing followed by strictly
        increasing) function with a discrete domain, but it is allowed to be
        constant in between the two strictly monotonic parts. In the latter
        case, the smallest input that is a minimum will be returned.
    `lower` is an inclusive lower bound for the target input.
    `upper` is an inclusive upper bound for the target input.
    Making your function `f` cache its results is highly recommended, as this
        algorithm may call it multiple times for the same input.
    """
    while upper - lower > 2:
        m1 = (2 * lower + upper) // 3
        m2 = (lower + 2 * upper) // 3
        if f(m1) >= f(m2):
            lower = m1
        else:
            upper = m2
    if f(lower) <= f(lower + 1):
        return lower
    if f(upper - 1) <= f(upper):
        return upper - 1
    return upper


def ternary_search_continuous_func(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    abs_tol: float = 10**-6,
    rel_tol: float = 10**-6,
) -> float:
    """
    Calculates the smallest input satisfying:
        `f(input - abs_tol) >= min(f) <= f(input + abs_tol)` or
        `f(input - input*rel_tol) >= min(f) <= f(input + input*rel_tol)`
    `f` must be a unimodal (strictly decreasing followed by strictly
        increasing) function with a continuous domain, but it is allowed to be
        constant in between the two strictly monotonic parts. In the latter
        case, any input that is a minimum could be returned.
    `lower` and `upper` are the bounds in between which the minimum resides.
        Whether they are inclusive or exclusive doesn't matter.
    """
    while upper - lower >= abs_tol and abs((upper - lower) / upper) >= rel_tol:
        m1 = (2 * lower + upper) / 3
        m2 = (lower + 2 * upper) / 3
        if f(m1) >= f(m2):
            lower = m1
        else:
            upper = m2
    return (lower + upper) / 2


class SegTree:
    """
    A segment tree is a binary tree that makes associative operations between
    multiple consecutive elements in a list faster. The bottom layer is the
    original list, and any node above the bottom layer contains the sum of the
    two nodes beneath.
    """

    def __init__(
        self, l: list[float] | None = None, length: int | None = None
    ):
        """
        Creates a segment tree. If `l` is given, `length` is ignored, and we
        build a segment tree with underlying list `l`. If no list is given,
        `length` must be given, and we build a segment tree with an underlying
        all-zeros list of that length.
        """
        if l is not None:
            self.len = len(l)
            self.t = [0.0] * (2 * self.len)
            self.t[self.len :] = l
            for i in reversed(range(1, self.len)):
                self.t[i] = self.t[i * 2] + self.t[i * 2 + 1]
        elif length is not None:
            self.len = length
            self.t = [0.0] * (2 * self.len)

    def modify(self, idx: int, val: float):
        """Set the value at index `idx` to `val`."""
        idx += self.len
        self.t[idx] = val
        while idx > 1:
            self.t[idx // 2] = self.t[idx] + self.t[idx ^ 1]
            idx //= 2

    def query(self, l: int, r: int) -> float:
        """Get the sum of all values in the range [`l`, `r`)."""
        res = 0
        l += self.len
        r += self.len
        while l < r:
            if l & 1:
                res += self.t[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.t[r]
            l //= 2
            r //= 2
        return res


# ############################### NUMBER THEORY ###############################
# Use `gcd` from the math library if you're only interested in the gcd! It is
# faster than an iterative implementation in Python.


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Calculate (gcd, s, t) in the equation a*s + b*t = gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - r * q
        old_s, s = s, old_s - s * q
    t = (old_r - old_s * a) // b if b else 0
    return old_r, old_s, t


def prime_factors(n: int) -> Iterator[int]:
    """Generate all prime factors of the number n in O(sqrt(n)) time."""
    from math import sqrt

    while n % 2 == 0:
        yield 2
        n //= 2
    for i in range(3, int(sqrt(n)) + 1, 2):
        while n % i == 0:
            yield i
            n //= i
        if n == 1:
            return
    if n > 2:
        yield n


def factors(n: int) -> Iterable[int]:
    """Generate all factors of the number n in O(sqrt(n)) time."""
    pfactors = prime_factors(n)
    factors = {1}
    for p in pfactors:
        factors.update({p * f for f in factors})
    return factors


def check_primality(n: int) -> list[bool]:
    """Check the primality of every number up to n-1 in O(n) time."""
    from math import sqrt

    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(4, int(sqrt(n)) + 1, 2):
        is_prime[i] = False
    for i in range(3, int(sqrt(n)) + 1, 2):
        if is_prime[i]:  # use the Sieve of Eratosthenes
            for j in range(i**2, n, i):
                is_prime[j] = False
    return is_prime


def interp1d(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Return an interpolated value y given two points and a value x."""
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of a and b."""
    from math import gcd

    return a // gcd(a, b) * b


def pascal(n: int) -> list[int]:
    """Returns the nth row of Pascal's triangle in O(n) time."""
    row = [1] * (n + 1)
    for k in range(n // 2):
        x = row[k] * (n - k) // (k + 1)
        row[k + 1] = x
        row[n - k - 1] = x
    return row
