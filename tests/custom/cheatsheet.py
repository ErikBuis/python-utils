# pyright: reportPrivateUsage=false
# pyright: reportUninitializedInstanceVariable=false


import unittest
from collections.abc import Callable
from math import inf
from typing import TypeVar

from typing_extensions import override

from python_utils.custom import cheatsheet


T = TypeVar("T")


def create_undigraph() -> list[list[int]]:
    return [[1, 2, 3, 4], [0, 2], [0, 1], [0], [0], [6], [5]]


def create_digraph() -> list[list[int]]:
    return [[1, 4], [2, 3], [3], [1, 5], [5], [6], [4]]


def create_dag() -> list[list[int]]:
    return [[2, 5], [3, 4], [3, 5], [5, 6], [5], [6], []]


def create_tree() -> list[list[int]]:
    return [[1, 2, 3], [4, 5], [6], [], [], [], []]


def create_weighted_undigraph() -> list[list[tuple[int, float]]]:
    return [
        [(1, 1), (2, 3), (3, 2), (4, 4)],
        [(0, 1), (2, 1)],
        [(0, 3), (1, 1)],
        [(0, 2)],
        [(0, 4)],
        [(6, 1)],
        [(5, 1)],
    ]


def create_weighted_digraph() -> list[list[tuple[int, float]]]:
    return [
        [(1, 1), (4, 4)],
        [(2, 2), (3, 4)],
        [(3, 1)],
        [(1, 3), (5, 1)],
        [(5, 2)],
        [(6, 1)],
        [(4, 1)],
    ]


def create_weighted_dag() -> list[list[tuple[int, float]]]:
    return [
        [(2, 1), (5, 5)],
        [(3, 5), (4, 2)],
        [(3, 1), (5, 3)],
        [(5, 1), (6, 3)],
        [(5, 3)],
        [(6, 1)],
        [],
    ]


def create_weighted_tree() -> list[list[tuple[int, float]]]:
    return [
        [(1, 1), (2, 2), (3, 3)],
        [(4, 4), (5, 5)],
        [(6, 6)],
        [],
        [],
        [],
        [],
    ]


def dict_to_list(d: dict[int, T], V: int, default: T) -> list[T]:
    l = [default for _ in range(V)]
    for v, t in d.items():
        l[v] = t
    return l


def list_to_dict(l: list[T]) -> dict[int, T]:
    d = {}
    for v, t in enumerate(l):
        d[v] = t
    return d


def unweighted_to_weighted(
    adjlist: list[list[int]],
) -> list[list[tuple[int, float]]]:
    return [[(v, 1) for v in nbrs] for nbrs in adjlist]


def weighted_to_unweighted(
    adjlist: list[list[tuple[int, float]]],
) -> list[list[int]]:
    return [[v for v, _ in nbrs] for nbrs in adjlist]


def check_topological_ordering(
    self: unittest.TestCase, adjlist: list[list[int]], toposort: list[int]
) -> None:
    # Some preprocessing to make the assertions easier.
    toposort2idx = [-1] * len(toposort)
    for i, v in enumerate(toposort):
        toposort2idx[v] = i

    # Check that the topological ordering is correct.
    for u, nbrs in enumerate(adjlist):
        for v in nbrs:
            self.assertLess(toposort2idx[u], toposort2idx[v])


def edge_count_heuristic(
    adjlist: list[list[tuple[int, float]]], goal: int
) -> Callable[[int], float]:
    # This is always an admissible heuristic if weight >= 1 for all edges.
    adjlist_unweighted = weighted_to_unweighted(adjlist)

    def heuristic(v: int) -> float:
        dist, _ = cheatsheet.bfs(adjlist_unweighted, v)
        return dist[goal]

    return heuristic


class TestReachableVertices(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()

    def test_reachable_vertices_undigraph(self) -> None:
        reachable = cheatsheet.reachable_vertices(self.undigraph, 0)
        self.assertSetEqual(reachable, {0, 1, 2, 3, 4})

        reachable = cheatsheet.reachable_vertices(self.undigraph, 3)
        self.assertSetEqual(reachable, {0, 1, 2, 3, 4})

        reachable = cheatsheet.reachable_vertices(self.undigraph, 5)
        self.assertSetEqual(reachable, {5, 6})

        reachable = cheatsheet.reachable_vertices(self.undigraph, 6)
        self.assertSetEqual(reachable, {5, 6})

    def test_reachable_vertices_digraph(self) -> None:
        reachable = cheatsheet.reachable_vertices(self.digraph, 0)
        self.assertSetEqual(reachable, {0, 1, 2, 3, 4, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.digraph, 3)
        self.assertSetEqual(reachable, {1, 2, 3, 4, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.digraph, 5)
        self.assertSetEqual(reachable, {4, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.digraph, 6)
        self.assertSetEqual(reachable, {4, 5, 6})

    def test_reachable_vertices_dag(self) -> None:
        reachable = cheatsheet.reachable_vertices(self.dag, 0)
        self.assertSetEqual(reachable, {0, 2, 3, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.dag, 1)
        self.assertSetEqual(reachable, {1, 3, 4, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.dag, 3)
        self.assertSetEqual(reachable, {3, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.dag, 6)
        self.assertSetEqual(reachable, {6})

    def test_reachable_vertices_tree(self) -> None:
        reachable = cheatsheet.reachable_vertices(self.tree, 0)
        self.assertSetEqual(reachable, {0, 1, 2, 3, 4, 5, 6})

        reachable = cheatsheet.reachable_vertices(self.tree, 1)
        self.assertSetEqual(reachable, {1, 4, 5})

        reachable = cheatsheet.reachable_vertices(self.tree, 2)
        self.assertSetEqual(reachable, {2, 6})

        reachable = cheatsheet.reachable_vertices(self.tree, 5)
        self.assertSetEqual(reachable, {5})


class TestConnectedComponents(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()

    def test_connected_components_undigraph(self) -> None:
        components = cheatsheet.connected_components(self.undigraph)
        self.assertListEqual(components, [{0, 1, 2, 3, 4}, {5, 6}])


class TestBFS(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()

    def test_bfs_undigraph(self) -> None:
        dist, parent = cheatsheet.bfs(self.undigraph, 0)
        self.assertListEqual(dist, [0, 1, 1, 1, 1, inf, inf])
        self.assertListEqual(parent, [-1, 0, 0, 0, 0, -1, -1])

        dist, parent = cheatsheet.bfs(self.undigraph, 3)
        self.assertListEqual(dist, [1, 2, 2, 0, 2, inf, inf])
        self.assertListEqual(parent, [3, 0, 0, -1, 0, -1, -1])

        dist, parent = cheatsheet.bfs(self.undigraph, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 0, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, 5])

        dist, parent = cheatsheet.bfs(self.undigraph, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 1, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, 6, -1])

    def test_bfs_digraph(self) -> None:
        dist, parent = cheatsheet.bfs(self.digraph, 0)
        self.assertListEqual(dist, [0, 1, 2, 2, 1, 2, 3])
        self.assertListEqual(parent, [-1, 0, 1, 1, 0, 4, 5])

        dist, parent = cheatsheet.bfs(self.digraph, 3)
        self.assertListEqual(dist, [inf, 1, 2, 0, 3, 1, 2])
        self.assertListEqual(parent, [-1, 3, 1, -1, 6, 3, 5])

        dist, parent = cheatsheet.bfs(self.digraph, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, 2, 0, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, 6, -1, 5])

        dist, parent = cheatsheet.bfs(self.digraph, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, 1, 2, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, 6, 4, -1])

    def test_bfs_dag(self) -> None:
        dist, parent = cheatsheet.bfs(self.dag, 0)
        self.assertListEqual(dist, [0, inf, 1, 2, inf, 1, 2])
        self.assertListEqual(parent, [-1, -1, 0, 2, -1, 0, 5])

        dist, parent = cheatsheet.bfs(self.dag, 1)
        self.assertListEqual(dist, [inf, 0, inf, 1, 1, 2, 2])
        self.assertTrue(
            parent == [-1, -1, -1, 1, 1, 3, 3]
            or parent == [-1, -1, -1, 1, 1, 4, 3]
        )

        dist, parent = cheatsheet.bfs(self.dag, 3)
        self.assertListEqual(dist, [inf, inf, inf, 0, inf, 1, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, 3, 3])

        dist, parent = cheatsheet.bfs(self.dag, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, inf, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, -1])

    def test_bfs_tree(self) -> None:
        dist, parent = cheatsheet.bfs(self.tree, 0)
        self.assertListEqual(dist, [0, 1, 1, 1, 2, 2, 2])
        self.assertListEqual(parent, [-1, 0, 0, 0, 1, 1, 2])

        dist, parent = cheatsheet.bfs(self.tree, 1)
        self.assertListEqual(dist, [inf, 0, inf, inf, 1, 1, inf])
        self.assertListEqual(parent, [-1, -1, -1, -1, 1, 1, -1])

        dist, parent = cheatsheet.bfs(self.tree, 2)
        self.assertListEqual(dist, [inf, inf, 0, inf, inf, inf, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, 2])

        dist, parent = cheatsheet.bfs(self.tree, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 0, inf])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, -1])


class TestBFSVType(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()

    def compare_to_bfs(self, adjlist: list[list[int]], start: int) -> None:
        move = adjlist.__getitem__

        dist1, parent1 = cheatsheet.bfs(adjlist, start)
        dist_dict, parent_dict = cheatsheet.bfs_v_type(move, start)

        dist2 = dict_to_list(dist_dict, len(adjlist), inf)
        parent2 = dict_to_list(parent_dict, len(adjlist), -1)

        self.assertListEqual(dist1, dist2)
        self.assertListEqual(parent1, parent2)

    def test_bfs_v_type_undigraph(self) -> None:
        for start in (0, 3, 5, 6):
            self.compare_to_bfs(self.undigraph, start)

    def test_bfs_v_type_digraph(self) -> None:
        for start in (0, 3, 5, 6):
            self.compare_to_bfs(self.digraph, start)

    def test_bfs_v_type_dag(self) -> None:
        for start in (0, 1, 3, 6):
            self.compare_to_bfs(self.dag, start)

    def test_bfs_v_type_tree(self) -> None:
        for start in (0, 1, 2, 5):
            self.compare_to_bfs(self.tree, start)


class TestReverseDigraph(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()

    def test_reverse_undigraph(self) -> None:
        reversed_graph = cheatsheet.reverse_digraph(self.undigraph)
        self.assertListEqual(reversed_graph, self.undigraph)

    def test_reverse_digraph(self) -> None:
        reversed_graph = cheatsheet.reverse_digraph(self.digraph)
        self.assertListEqual(
            reversed_graph, [[], [0, 3], [1], [1, 2], [0, 6], [3, 4], [5]]
        )

    def test_reverse_dag(self) -> None:
        reversed_graph = cheatsheet.reverse_digraph(self.dag)
        self.assertListEqual(
            reversed_graph, [[], [], [0], [1, 2], [1], [0, 2, 3, 4], [3, 5]]
        )

    def test_reverse_tree(self) -> None:
        reversed_graph = cheatsheet.reverse_digraph(self.tree)
        self.assertListEqual(
            reversed_graph, [[], [0], [0], [0], [1], [1], [2]]
        )


class TestKahn(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()
        self.undigraph_rev = cheatsheet.reverse_digraph(self.undigraph)
        self.digraph_rev = cheatsheet.reverse_digraph(self.digraph)
        self.dag_rev = cheatsheet.reverse_digraph(self.dag)
        self.tree_rev = cheatsheet.reverse_digraph(self.tree)

    def check_no_topological_ordering(
        self, adjlist: list[list[int]], toposort: list[int]
    ) -> None:
        # Check that no topological ordering was returned.
        self.assertLess(len(toposort), len(adjlist))

    def test_kahn_undigraph(self) -> None:
        toposort = cheatsheet.kahn(self.undigraph, self.undigraph_rev)
        self.check_no_topological_ordering(self.undigraph, toposort)

    def test_kahn_digraph(self) -> None:
        toposort = cheatsheet.kahn(self.digraph, self.digraph_rev)
        self.check_no_topological_ordering(self.digraph, toposort)

    def test_kahn_dag(self) -> None:
        toposort = cheatsheet.kahn(self.dag, self.dag_rev)
        check_topological_ordering(self, self.dag, toposort)

    def test_kahn_tree(self) -> None:
        toposort = cheatsheet.kahn(self.tree, self.tree_rev)
        check_topological_ordering(self, self.tree, toposort)


class TestTarjan(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()

    def check_no_sccs(self, adjlist: list[list[int]]) -> None:
        # Check that there are no SCCs.
        amount_components, components = cheatsheet.tarjan(adjlist)
        self.assertEqual(amount_components, len(adjlist))

        # Create a toposort from the components.
        toposort_tups = sorted(
            enumerate(components), key=lambda x: x[1], reverse=True
        )
        toposort = [tup[0] for tup in toposort_tups]

        # Check that the topological ordering is correct.
        check_topological_ordering(self, adjlist, toposort)

    def test_tarjan_undigraph(self) -> None:
        amount_components, components = cheatsheet.tarjan(self.undigraph)
        self.assertEqual(amount_components, 2)
        self.assertTrue(
            components == [0, 0, 0, 0, 0, 1, 1]
            or components == [1, 1, 1, 1, 1, 0, 0]
        )

    def test_tarjan_digraph(self) -> None:
        amount_components, components = cheatsheet.tarjan(self.digraph)
        self.assertEqual(amount_components, 3)
        self.assertListEqual(components, [2, 1, 1, 1, 0, 0, 0])

    def test_tarjan_dag(self) -> None:
        self.check_no_sccs(self.dag)

    def test_tarjan_tree(self) -> None:
        self.check_no_sccs(self.tree)


class TestComponentGraph(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()
        self.undigraph_amount_components, self.undigraph_components = (
            cheatsheet.tarjan(self.undigraph)
        )
        self.digraph_amount_components, self.digraph_components = (
            cheatsheet.tarjan(self.digraph)
        )
        self.dag_amount_components, self.dag_components = cheatsheet.tarjan(
            self.dag
        )
        self.tree_amount_components, self.tree_components = cheatsheet.tarjan(
            self.tree
        )

    def check_no_sccs(
        self,
        adjlist: list[list[int]],
        amount_components: int,
        components: list[int],
        component_adjlist: list[list[int]],
    ) -> None:
        # Check that there are no SCCs.
        self.assertEqual(amount_components, len(adjlist))

        # Check that exactly every edge in the original graph is also in the
        # component graph, and that no other edges are in the component graph.
        for u, nbrs in enumerate(adjlist):
            self.assertSetEqual(
                set(component_adjlist[components[u]]),
                {components[v] for v in nbrs},
            )

    def test_component_graph_undigraph(self) -> None:
        component_adjlist = cheatsheet.component_graph(
            self.undigraph,
            self.undigraph_amount_components,
            self.undigraph_components,
        )
        self.assertListEqual(component_adjlist, [[], []])

    def test_component_graph_digraph(self) -> None:
        component_adjlist = cheatsheet.component_graph(
            self.digraph,
            self.digraph_amount_components,
            self.digraph_components,
        )
        self.assertListEqual(component_adjlist, [[], [0], [0, 1]])

    def test_component_graph_dag(self) -> None:
        component_adjlist = cheatsheet.component_graph(
            self.dag, self.dag_amount_components, self.dag_components
        )
        self.check_no_sccs(
            self.dag,
            self.dag_amount_components,
            self.dag_components,
            component_adjlist,
        )

    def test_component_graph_tree(self) -> None:
        component_adjlist = cheatsheet.component_graph(
            self.tree, self.tree_amount_components, self.tree_components
        )
        self.check_no_sccs(
            self.tree,
            self.tree_amount_components,
            self.tree_components,
            component_adjlist,
        )


class TestDijkstra(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.undigraph = create_undigraph()
        self.digraph = create_digraph()
        self.dag = create_dag()
        self.tree = create_tree()
        self.weighted_undigraph = create_weighted_undigraph()
        self.weighted_digraph = create_weighted_digraph()
        self.weighted_dag = create_weighted_dag()
        self.weighted_tree = create_weighted_tree()

    def compare_to_bfs(
        self,
        adjlist: list[list[int]],
        adjlist_weighted: list[list[tuple[int, float]]],
        start: int,
    ) -> None:
        dist1, parent1 = cheatsheet.bfs(adjlist, start)
        dist2, parent2 = cheatsheet.dijkstra(adjlist_weighted, start)
        self.assertListEqual(dist1, dist2)
        self.assertListEqual(parent1, parent2)

    def test_dijkstra_undigraph_compare_bfs(self) -> None:
        undigraph_weighted = unweighted_to_weighted(self.undigraph)
        for start in (0, 3, 5, 6):
            self.compare_to_bfs(self.undigraph, undigraph_weighted, start)

    def test_dijkstra_digraph_compare_bfs(self) -> None:
        digraph_weighted = unweighted_to_weighted(self.digraph)
        for start in (0, 3, 5, 6):
            self.compare_to_bfs(self.digraph, digraph_weighted, start)

    def test_dijkstra_dag_compare_bfs(self) -> None:
        dag_weighted = unweighted_to_weighted(self.dag)
        for start in (0, 1, 3, 6):
            self.compare_to_bfs(self.dag, dag_weighted, start)

    def test_dijkstra_tree_compare_bfs(self) -> None:
        tree_weighted = unweighted_to_weighted(self.tree)
        for start in (0, 1, 2, 5):
            self.compare_to_bfs(self.tree, tree_weighted, start)

    def test_dijkstra_weighted_undigraph(self) -> None:
        dist, parent = cheatsheet.dijkstra(self.weighted_undigraph, 0)
        self.assertListEqual(dist, [0, 1, 2, 2, 4, inf, inf])
        self.assertListEqual(parent, [-1, 0, 1, 0, 0, -1, -1])

        dist, parent = cheatsheet.dijkstra(self.weighted_undigraph, 3)
        self.assertListEqual(dist, [2, 3, 4, 0, 6, inf, inf])
        self.assertListEqual(parent, [3, 0, 1, -1, 0, -1, -1])

        dist, parent = cheatsheet.dijkstra(self.weighted_undigraph, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 0, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_undigraph, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 1, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, 6, -1])

    def test_dijkstra_weighted_digraph(self) -> None:
        dist, parent = cheatsheet.dijkstra(self.weighted_digraph, 0)
        self.assertListEqual(dist, [0, 1, 3, 4, 4, 5, 6])
        self.assertListEqual(parent, [-1, 0, 1, 2, 0, 3, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_digraph, 3)
        self.assertListEqual(dist, [inf, 3, 5, 0, 3, 1, 2])
        self.assertListEqual(parent, [-1, 3, 1, -1, 6, 3, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_digraph, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, 2, 0, 1])
        self.assertListEqual(parent, [-1, -1, -1, -1, 6, -1, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_digraph, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, 1, 3, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, 6, 4, -1])

    def test_dijkstra_weighted_dag(self) -> None:
        dist, parent = cheatsheet.dijkstra(self.weighted_dag, 0)
        self.assertListEqual(dist, [0, inf, 1, 2, inf, 3, 4])
        self.assertListEqual(parent, [-1, -1, 0, 2, -1, 3, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_dag, 1)
        self.assertListEqual(dist, [inf, 0, inf, 5, 2, 5, 6])
        self.assertListEqual(parent, [-1, -1, -1, 1, 1, 4, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_dag, 3)
        self.assertListEqual(dist, [inf, inf, inf, 0, inf, 1, 2])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, 3, 5])

        dist, parent = cheatsheet.dijkstra(self.weighted_dag, 6)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, inf, 0])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, -1])

    def test_dijkstra_weighted_tree(self) -> None:
        dist, parent = cheatsheet.dijkstra(self.weighted_tree, 0)
        self.assertListEqual(dist, [0, 1, 2, 3, 5, 6, 8])
        self.assertListEqual(parent, [-1, 0, 0, 0, 1, 1, 2])

        dist, parent = cheatsheet.dijkstra(self.weighted_tree, 1)
        self.assertListEqual(dist, [inf, 0, inf, inf, 4, 5, inf])
        self.assertListEqual(parent, [-1, -1, -1, -1, 1, 1, -1])

        dist, parent = cheatsheet.dijkstra(self.weighted_tree, 2)
        self.assertListEqual(dist, [inf, inf, 0, inf, inf, inf, 6])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, 2])

        dist, parent = cheatsheet.dijkstra(self.weighted_tree, 5)
        self.assertListEqual(dist, [inf, inf, inf, inf, inf, 0, inf])
        self.assertListEqual(parent, [-1, -1, -1, -1, -1, -1, -1])


class TestDijkstraVType(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.weighted_undigraph = create_weighted_undigraph()
        self.weighted_digraph = create_weighted_digraph()
        self.weighted_dag = create_weighted_dag()
        self.weighted_tree = create_weighted_tree()

    def compare_to_dijkstra(
        self, adjlist: list[list[tuple[int, float]]], start: int
    ) -> None:
        move = adjlist.__getitem__

        dist1, parent1 = cheatsheet.dijkstra(adjlist, start)
        dist_dict, parent_dict = cheatsheet.dijkstra_v_type(move, start)

        dist2 = dict_to_list(dist_dict, len(adjlist), inf)
        parent2 = dict_to_list(parent_dict, len(adjlist), -1)

        self.assertListEqual(dist1, dist2)
        self.assertListEqual(parent1, parent2)

    def test_dijkstra_v_type_undigraph_compare_dijkstra(self) -> None:
        for start in (0, 3, 5, 6):
            self.compare_to_dijkstra(self.weighted_undigraph, start)

    def test_dijkstra_v_type_digraph_compare_dijkstra(self) -> None:
        for start in (0, 3, 5, 6):
            self.compare_to_dijkstra(self.weighted_digraph, start)

    def test_dijkstra_v_type_dag_compare_dijkstra(self) -> None:
        for start in (0, 1, 3, 6):
            self.compare_to_dijkstra(self.weighted_dag, start)

    def test_dijkstra_v_type_tree_compare_dijkstra(self) -> None:
        for start in (0, 1, 2, 5):
            self.compare_to_dijkstra(self.weighted_tree, start)


class TestAStar(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.weighted_undigraph = create_weighted_undigraph()
        self.weighted_digraph = create_weighted_digraph()
        self.weighted_dag = create_weighted_dag()
        self.weighted_tree = create_weighted_tree()

    def compare_to_dijkstra(
        self, adjlist: list[list[tuple[int, float]]], start: int, goal: int
    ) -> None:
        # Run Dijkstra's algorithm.
        dist, parent = cheatsheet.dijkstra(adjlist, start)
        cost1 = dist[goal]
        path1 = [goal]
        while parent[path1[-1]] != -1:
            path1.append(parent[path1[-1]])
        path1.reverse()

        # Run A*.
        cost2, path2 = cheatsheet.a_star(adjlist, start, goal, lambda _: 0)

        # Check that the results are the same.
        self.assertEqual(cost1, cost2)
        self.assertListEqual(path1, path2)

    def check_heuristics(
        self, adjlist: list[list[tuple[int, float]]], start: int, goal: int
    ) -> None:
        heuristic = edge_count_heuristic(adjlist, goal)

        cost1, path1 = cheatsheet.a_star(adjlist, start, goal, lambda _: 0)
        cost2, path2 = cheatsheet.a_star(adjlist, start, goal, heuristic)

        self.assertEqual(cost1, cost2)
        self.assertListEqual(path1, path2)

    def test_a_star_weighted_undigraph_compare_dijkstra(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_dijkstra(self.weighted_undigraph, start, goal)

    def test_a_star_weighted_digraph_compare_dijkstra(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_dijkstra(self.weighted_digraph, start, goal)

    def test_a_star_weighted_dag_compare_dijkstra(self) -> None:
        for start in (0, 1, 3, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_dijkstra(self.weighted_dag, start, goal)

    def test_a_star_weighted_tree_compare_dijkstra(self) -> None:
        for start in (0, 1, 2, 5):
            for goal in (1, 2, 4, 6):
                self.compare_to_dijkstra(self.weighted_tree, start, goal)

    def test_a_star_weighted_undigraph_check_heuristics(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.check_heuristics(self.weighted_undigraph, start, goal)

    def test_a_star_weighted_digraph_check_heuristics(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.check_heuristics(self.weighted_digraph, start, goal)

    def test_a_star_weighted_dag_check_heuristics(self) -> None:
        for start in (0, 1, 3, 6):
            for goal in (1, 2, 4, 6):
                self.check_heuristics(self.weighted_dag, start, goal)

    def test_a_star_weighted_tree_check_heuristics(self) -> None:
        for start in (0, 1, 2, 5):
            for goal in (1, 2, 4, 6):
                self.check_heuristics(self.weighted_tree, start, goal)


class TestAStarVType(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.weighted_undigraph = create_weighted_undigraph()
        self.weighted_digraph = create_weighted_digraph()
        self.weighted_dag = create_weighted_dag()
        self.weighted_tree = create_weighted_tree()

    def compare_to_a_star(
        self, adjlist: list[list[tuple[int, float]]], start: int, goal: int
    ) -> None:
        move = adjlist.__getitem__
        heuristic = edge_count_heuristic(adjlist, goal)

        cost1, path1 = cheatsheet.a_star(adjlist, start, goal, heuristic)
        cost2, path2 = cheatsheet.a_star_v_type(move, start, goal, heuristic)

        self.assertEqual(cost1, cost2)
        self.assertListEqual(path1, path2)

    def test_a_star_v_type_weighted_undigraph(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_a_star(self.weighted_undigraph, start, goal)

    def test_a_star_v_type_weighted_digraph(self) -> None:
        for start in (0, 3, 5, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_a_star(self.weighted_digraph, start, goal)

    def test_a_star_v_type_weighted_dag(self) -> None:
        for start in (0, 1, 3, 6):
            for goal in (1, 2, 4, 6):
                self.compare_to_a_star(self.weighted_dag, start, goal)

    def test_a_star_v_type_weighted_tree(self) -> None:
        for start in (0, 1, 2, 5):
            for goal in (1, 2, 4, 6):
                self.compare_to_a_star(self.weighted_tree, start, goal)


class TestPrim(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.weighted_undigraph = create_weighted_undigraph()

    def test_prim_weighted_undigraph_start_0(self) -> None:
        mst = cheatsheet.prim(self.weighted_undigraph, 0)
        self.assertListEqual(
            mst,
            [
                [(1, 1), (3, 2), (4, 4)],
                [(0, 1), (2, 1)],
                [(1, 1)],
                [(0, 2)],
                [(0, 4)],
                [],
                [],
            ],
        )

    def test_prim_weighted_undigraph_start_5(self) -> None:
        mst = cheatsheet.prim(self.weighted_undigraph, 5)
        self.assertListEqual(mst, [[], [], [], [], [], [(6, 1)], [(5, 1)]])


if __name__ == "__main__":
    unittest.main()
