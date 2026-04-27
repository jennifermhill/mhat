import numpy as np
import ilpy
import motile
from motile.variables import EdgeSelected


class EdgePairs(motile.variables.Variable):

  @staticmethod
  def instantiate(solver):

      edge_pairs = [
          (in_edge, out_edge)
          # for each node
          for node in solver.graph.nodes
          # for each pair of incoming and outgoing edge (simple edges only)
          for in_edge in solver.graph.prev_edges[node]
          if isinstance(in_edge[0], (int, np.integer))
          for out_edge in solver.graph.next_edges[node]
          if isinstance(out_edge[0], (int, np.integer))
      ]

      return edge_pairs

  @staticmethod
  def instantiate_constraints(solver):

      edge_indicators = solver.get_variables(EdgeSelected)
      edge_pair_indicators = solver.get_variables(EdgePairs)

      constraints = []
      for (in_edge, out_edge), pair_index in edge_pair_indicators.items():

          in_edge_index = edge_indicators[in_edge]
          out_edge_index = edge_indicators[out_edge]

          # edge pair indicator = 1 <=> in edge = 1 and out edge = 1
          yield pair_index * 2 + in_edge_index * -1 + out_edge_index * -1 <= 0
          yield pair_index * -1 + out_edge_index + in_edge_index <= 1
          

class CurvatureCost(motile.costs.Cost):

    def __init__(self, position_attribute, weight=1.0, constant=0.0):
        self.position_attribute = position_attribute
        self.weight = motile.costs.Weight(weight)
        self.constant = motile.costs.Weight(constant)

    def apply(self, solver):

        # get edge pair variables
        edge_pair_indicators = solver.get_variables(EdgePairs)
        print(f"Applying curvature cost to {len(edge_pair_indicators)} edge pairs")

        for (in_edge, out_edge), index in edge_pair_indicators.items():

            in_offset = self.get_edge_offset(solver.graph, in_edge)
            out_offset = self.get_edge_offset(solver.graph, out_edge)

            curvature_cost = np.linalg.norm(out_offset - in_offset)

            solver.add_variable_cost(index, curvature_cost, self.weight)
            solver.add_variable_cost(index, 1.0, self.constant)

    def get_edge_offset(self, graph, edge):

        pos_v = np.array(graph.nodes[edge[1]][self.position_attribute])
        pos_u = np.array(graph.nodes[edge[0]][self.position_attribute])

        return pos_v - pos_u