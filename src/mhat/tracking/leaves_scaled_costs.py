from motile.costs import NodeSelection
from motile.variables import NodeSelected


class LeavesScaledNodeSelection(NodeSelection):
    """NodeSelection where each node's feature contribution is scaled by num_leaves.

    Used for cohesion/adhesion costs so that a merged candidate node contributes
    as if it represents `num_leaves` underlying fragments. This bakes the scaling
    into the feature matrix so SSVM weight fitting and ILP inference see the
    same effective costs.
    """

    def apply(self, solver):
        node_vars = solver.get_variables(NodeSelected)
        for node, index in node_vars.items():
            num_leaves = solver.graph.nodes[node].get("num_leaves", 1)
            attr_value = solver.graph.nodes[node][self.attribute]
            solver.add_variable_cost(index, num_leaves * attr_value, self.weight)
            solver.add_variable_cost(index, num_leaves * 1.0, self.constant)
