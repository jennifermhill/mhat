import motile
from motile.variables import EdgeSelected


class DivisionCost(motile.costs.Cost):
    """Cost charged to division hyperedges, via a 0/1 attribute on every edge.

    This exists instead of a plain EdgeSelection because EdgeSelection wraps both
    its weight and its constant in Weight objects, and Solver.add_cost registers
    every Weight it finds as a learnable parameter. That constant is charged to
    each selected edge regardless of the attribute value, so fitting it would
    learn a global per-edge bias rather than anything about divisions. Exposing
    only a weight keeps the fitted parameter meaningful and leaves nothing for
    the inference path to silently drop.

    Args:
        weight: Cost charged per selected division hyperedge.
        attribute: Edge attribute holding the 0/1 division indicator
            (see utils.add_division_attr).
    """

    def __init__(self, weight: float = 0.0, attribute: str = "is_division"):
        self.weight = motile.costs.Weight(weight)
        self.attribute = attribute

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)

        for edge, index in edge_variables.items():
            solver.add_variable_cost(
                index, solver.graph.edges[edge][self.attribute], self.weight
            )
