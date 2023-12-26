"""
This file contains extra functions to be used for competative programming
problems. They aren't used as much as those in the main cheatsheet, so they are
described here separately.

For speed purposes, inputs are not checked for validity!

The imports present at the top of functions should be put at the top of your
file, they should not stay in the function itself.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


VertexType = TypeVar("VertexType")
BaseType = TypeVar("BaseType", int, float)


# Breadth-first search for variable vertex types. Could be useful in a variant
# where the search is ended early, since this would prevent you from building
# the whole adjacency list in advance.
# Why not to include: it is a variant of BFS, which is already included in the
# cheatsheet.
def bfs_v_type(
    move: Callable[[VertexType], Iterable[VertexType]], start: VertexType
) -> tuple[
    defaultdict[VertexType, float], defaultdict[VertexType, VertexType | None]
]:
    """Calculate the shortest distance from the start to all other vertices.

    If a vertex is unreachable, its distance will be set to infinity.

    This function supports:
    - Undirected unweighted graphs
    - Directed unweighted graphs

    For weighted graphs, use Dijkstra's algorithm.

    Complexity: O(V + E)

    Args:
        move: A function that takes a vertex v and returns an iterable of
            neighbours.
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
            Defaultdict mapping each vertex to the shortest distance from the
                start. Warning: If the vertex is unreachable, it will not be
                present in the dict, but when indexing it, the value returned
                is infinity.
            Defaultdict mapping each vertex to its parent in the shortest path.
                Warning: If the vertex is unreachable, it will not be present
                in the dict, but when indexing it, the value returned is None.
    """
    from collections import defaultdict, deque
    from math import inf

    dist: defaultdict[VertexType, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: defaultdict[VertexType, VertexType | None] = defaultdict(
        lambda: None
    )
    queue: deque[VertexType] = deque((start,))
    while queue:
        u = queue.popleft()
        for v in move(u):
            if v not in dist:  # we didn't visit this vertex yet
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


# Dijkstra's algorithm for variable vertex types. Could be useful in a variant
# where the search is ended early, since this would prevent you from building
# the whole adjacency list in advance.
# Why not to include: it is a variant of Dijkstra's algorithm, which is already
# included in the cheatsheet.
def dijkstra_v_type(
    move: Callable[[VertexType], Iterable[tuple[VertexType, float]]],
    start: VertexType,
) -> tuple[
    defaultdict[VertexType, float], defaultdict[VertexType, VertexType | None]
]:
    """Dijkstra's algorithm finds the shortest path from start to all vertices.

    If a vertex is unreachable, its distance will be set to infinity, and its
    parent will be set to -1.

    This function supports:
    - Undirected weighted graphs
    - Directed weighted graphs

    For unweighted graphs, use BFS instead.

    Complexity: O((V + E) * log(V))

    Args:
        move: A function that takes a vertex v and returns an iterable of
            (neighbour, weight) tuples.
        start: The vertex to start the search from.

    Returns:
        Tuple containing:
            Defaultdict mapping each vertex to the shortest distance from the
                start. Warning: If the vertex is unreachable, it will not be
                present in the dict, but when indexing it, the value returned
                is infinity.
            Defaultdict mapping each vertex to its parent in the shortest path.
                Warning: If the vertex is unreachable, it will not be present
                in the dict, but when indexing it, the value returned is None.
    """
    from collections import defaultdict
    from heapq import heappop, heappush
    from math import inf

    dist: defaultdict[VertexType, float] = defaultdict(lambda: inf)
    dist[start] = 0
    parent: defaultdict[VertexType, VertexType | None] = defaultdict(
        lambda: None
    )
    # Store the distance to each vertex to ensure the closest is popped first.
    prioq: list[tuple[float, VertexType]] = []  # (dist, vertex)
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


# A* algorithm for integer vertex types. Is usually slower than the A*
# algorithm that works with variable vertex types, since the search is usually
# ended early. In these cases, is more memory efficient to specify a move
# function than spacify the whole adjlist, since you don't have to build the
# whole adjacency list in advance.
# Why not to include: it is a variant of A*, which is already included in the
# cheatsheet.
def a_star(
    adjlist: list[list[tuple[int, float]]],
    start: int,
    goal: int,
    heuristic: Callable[[int], float],
) -> tuple[float, list[int]]:
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
        adjlist: The adjacency list of the graph.
            Length: V
            Length of inner lists: degree(v)
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


# Kruskal's algorithm.
# Why not to include: Prims algorithm is shorter but is often equally fast.
class UnionFind:
    """This data structure divides the vertices into clusters."""

    def __init__(self, V: int):
        """Initializes a union-find data structure."""
        self.parents = list(range(V))
        self.sizes = [1] * V

    def find(self, i: int) -> int:
        """Finds the root of the disjoint set that `i` is in."""
        if self.parents[i] == i:
            return i
        self.parents[i] = self.find(self.parents[i])
        return self.parents[i]

    def union(self, i: int, j: int) -> bool:
        """
        Unites the sets that `i` and `j` are in if they are disjoint.
        Returns whether they were in disjoint sets.
        """
        i_root, j_root = self.find(i), self.find(j)
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
    """
    Kruskals algorithm to compute a minimal spanning tree.
    `edge_list` is an undirected list of (vertex1, vertex2, weight) tuples.
    Returns a list of edges in the MST.
    """
    from operator import itemgetter

    # Sort edges by weight.
    edge_list.sort(key=itemgetter(2))
    # Continuously add the edge to your MST that doesn't result in a cycle.
    UF = UnionFind(V)
    return [(a, b, w) for a, b, w in edge_list if UF.union(a, b)]


# Binary search.
# Why not to include: the built-in `bisect` module is always faster.
def binary_search_index(sorted_l: list, value, rounddown: bool = True):
    """
    Return the index of `value` in a sorted list `sorted_l`.
    If `value` is not in `sorted_l`, the closest index smaller (if True) or
    bigger (if False) than the actual one will be returned instead. If `value`
    is smaller than the smallest value and this argument is True, or it is
    bigger than the biggest value and this argument is False, the lowest and
    highest indices are selected respectively.
    >>> binary_search_index([-6, -2.9, 4, 5, 7.2], 5)  # 3
    >>> binary_search_index([-6, -2.9, 4, 5, 7.2], 3)  # 1
    >>> binary_search_index([-6, -2.9, 4, 5, 7.2], 3, rounddown=False)  # 2
    >>> binary_search_index([-6, -2.9, 4, 5, 7.2], 8, rounddown=False)  # 4
    """
    # `lower` and `upper` are exclusive lower and upper bounds.
    lower, upper = -1, len(sorted_l)
    while upper - lower > 1:
        mid = (upper + lower) // 2
        mid_value = sorted_l[mid]
        if value == mid_value:
            return mid
        if value < mid_value:
            upper = mid
        else:
            lower = mid
    return max(0, lower) if rounddown else min(upper, len(sorted_l) - 1)


# Aho-Corasick's algorithm.
# Why not to include: it isn't used often and takes up a lot of space in the
# cheatsheet.
def aho_corasick(
    words: Iterable[str], target: str
) -> Iterator[tuple[str, int]]:
    """
    Aho-Corasick's algorithm finds all matches of all words in a target text.

    It returns tuples of (word, index_in_target) for each match found.
    `index_in_target` is the index in the target of the last letter in `word`.
    The matches are always returned in order of increasing index_in_target.

    If there are k words which in total consist of m characters, the target
    text consists of n characters and there are z matches, then the time
    complexity of this algorithm is O(n + m + z). Since we have to go through
    all characters and matches at least once, it is the theoretical limit in
    terms of efficiency.
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


# Hopcroft-Karp's algorithm.
# Why not to include: it can be replaced by Edmonds-Karp and takes up a lot of
# space in the cheatsheet.
def hopcroft_karp(
    V: int, adjlist: list[list[int]], left: list[bool]
) -> tuple[int, list[int]]:
    """
    Hopcroft-Karp's algorithm finds a maximal matching in an unweighted
    bipartite graph.

    `left` is a boolean array which gives the left part of the bipartite graph.
    If a vertex w is in adjlist[v], then precisely one of left[v] or left[w]
    should be True.

    Returns `(amount_matchings, matching)`, where `amount_matchings` is the
    amount of edges in the matching, and `matching` is a list containing the
    matched vertex for every vertex in the graph (where a -1 indicates an
    unmatched vertex).
    """
    from collections import deque

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


# Greatest common divisor.
# Why not to include: the built-in `gcd` from the math module is always faster.
def gcd(a: int, b: int) -> int:
    """Use Euclidean's algorithm to calculate the greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


# Chinese remainder theorem.
# Why not to include: it isn't used often enough.
def crt(*pairs: tuple[int, int]) -> tuple[int, int] | None:
    """
    The Chinese remainder theorem finds an integer x that satisfies:
        x = a1 (mod n1)
        ...
        x = am (mod nm)
    Return (x, lcm(n1, ..., nm)) if there is a solution, else None.
    """
    from cheatsheet import extended_gcd, lcm  # type: ignore

    a1, n1 = pairs[0]
    for a2, n2 in pairs[1:]:
        d, x_prime, _ = extended_gcd(n1, n2)
        if a1 % d != a2 % d:
            return
        a1 += x_prime * (a2 - a1) // d * n1
        n1 = lcm(n1, n2)
        a1 %= n1
    return a1, n1


def round_multiple(x: int | float, base: BaseType) -> BaseType:
    """Round a number to the nearest multiple of a base number."""
    return round(x / base) * base


def ceil_multiple(x: int | float, base: BaseType) -> BaseType:
    """Ceil a number to the nearest multiple of a base number."""
    from math import ceil

    return ceil(x / base) * base


def floor_multiple(x: int | float, base: BaseType) -> BaseType:
    """Floor a number to the nearest multiple of a base number."""
    # This is faster than floor(x // base) and int(x / base) and int(x // base)
    # for some reason, I tested it with timeit.
    from math import floor

    return floor(x / base) * base


# Explanation for the runtime of the factorization function:
# Why not to include: it is too long and not very useful.
# This function's runtime is bounded by the runtime of computing the prime
# factors. The runtime of actually finding the factors is much smaller,
# namely O[n^(1/k)], where k is an arbitrary constant.
# -------------------------------------------------------------------------
# Proof:
# 1. In the worst case, our number n is the product of the first p primes.
#    Now you might think that this would lead to a very inefficient
#    runtime, since we would have to try all combinations of these primes,
#    which would lead to O[2^p] runtime. If you thought this, you are
#    absolutely right! However, the key insight is that p will be much,
#    much smaller than n. The following bullet points will explain why. I
#    have ordered them in the same order as my thought process, as opposed
#    to the order in which they would be used in a formal proof.
# 2. The first thing you have to realize is that by definition, we have:
#    prod_{i=1}^p p_i = n
#    However, we don't have the values of the primes p_i. If we manage to
#    express these in terms of p and n, we would be able to express p in
#    terms of n!
# 3. Luckily, there exists something called the prime number theorem, which
#    states that the number of primes pi(m) <= m is approximately
#    m / log(m). Source:
#    https://en.wikipedia.org/wiki/Prime_number_theorem
#    This allows us to locate the i-th prime p_i, since according to our
#    approximation, when pi(m) = i, we will encounter p_i. Thus,
#    p_i / log(p_i) = i.
# 4. We now have an equation with only p's and n's. We could stop here,
#    but our current expression is not very helpful. After all, our goal
#    was to approximate the runtime using big-O notation, which we
#    currently cannot do. Therefore, we will make a few assumptions,
#    after which we will still end up with a runtime of O[n^(1/k)].  :D
# 5. Since we can't solve this equation analytically, we will overestimate
#    the amount of primes, since this will still result in a valid big-O
#    upper bound. Therefore, we will assume that the i-th prime is located
#    at p_i = i. This results in prod_{i=1}^p i = n, or p! = n.
# 6. We can approximate p! using Stirling's approximation, which states
#    that p! ~ sqrt(2*pi*p) * (p/e)^p. Source:
#    https://en.wikipedia.org/wiki/Stirling%27s_approximation
# 7. While this expression is nice, we still can't use it to approximate
#    the runtime. Therefore, we will make one final approximation, namely
#    that p! ~ (p/e)^p. The difference between the two is negligible, since
#    the first term grows far less quickly than the second for large p.
# 8. Now, we know that the runtime is O[2**p], but this is still not very
#    helpful. Therefore, we will simplify this expression by substituting
#    p with log2(q), resulting in O[2**p] = O[2**log2(q)] = O[q]. Let's
#    also calculate a new approximation for p!:
#    p! ~ (p/e)^p
#       = (log2(q)/e)^log2(q)
#       = log2(q)^log2(q) / e^log2(q)
#      in O[log2(q)^log2(q) * something smaller than a polynomial]
#       = O[log2(q)^log2(q)]
#       > O[any polynomial in q]   (since it is an exponential function)
# 9. Finally, we can express the runtime in terms of q:
#    O[n] = O[p!]
#         = O[log2(q)^log2(q)]
#         > O[q^k] for any k
#    O[q] < O[n^(1/k)] for any k
from cheatsheet import factors  # type: ignore # noqa: E402, F401
