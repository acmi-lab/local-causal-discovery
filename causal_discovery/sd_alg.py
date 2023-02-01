import numpy as np
import pandas as pd
import networkx as nx
import numpy
import pandas
import networkx
from itertools import combinations, permutations
import logging
from causallearn.utils.cit import CIT
from sklearn import linear_model
import collections
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import copy

from causal_discovery.utils import *
from causal_discovery.causal_discovery_base import SequentialLocalCausalDiscoveryBase


class SequentialDiscoveryAlgorithm(SequentialLocalCausalDiscoveryBase):
  """The Sequential Discovery (SD) algorithm runs the PC algorithm locally to 
  perform local structure learning for each sequential step.
  """

  def __init__(self, treatment_node="X", outcome_node="Y", alpha=0.05, 
               use_ci_oracle=True, graph_true=None, enable_logging=False,
               max_tests=None):
    
    def local_structure_fn(g, sep_set, main_node_id, data_matrix):
      return self._find_local_structure(g, sep_set, main_node_id, 
        data_matrix, self._partial_corr_independence_test)

    super(SequentialDiscoveryAlgorithm, self).__init__(treatment_node=treatment_node,
                                      outcome_node=outcome_node, alpha=alpha,
                                      use_ci_oracle=use_ci_oracle,
                                      graph_true=graph_true,
                                      enable_logging=enable_logging,
                                      max_tests=max_tests,
                                      local_structure_learning_func=local_structure_fn)

  def _find_local_structure(self, init_graph, sep_set_init, node, data_matrix, 
                            indep_test_func):
    nx = networkx

    alpha = self.alpha
    node_id_main = node
    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    if sep_set_init is not None:
      sep_set = sep_set_init
    else:
      sep_set = [[set() for i in range(node_size)] for j in range(node_size)]


    g = init_graph if init_graph is not None else self._create_complete_graph(node_ids)
    g = g.copy()

    g_old = g.copy()
    sep_set_old = copy.deepcopy(sep_set)

    l = 0
    while True:
      cont = False
      i = node_id_main
      for j in node_ids:
          if j == i:
            continue
            
          adj_i = list(g.neighbors(i))
          if j not in adj_i:
              continue
          else:
              adj_i.remove(j)
          
          
          if len(adj_i) >= l:
              adj_i_sorted = sorted(adj_i, key=lambda n: 0 if n == self.tmt_id else 1)
              for k in combinations(adj_i_sorted, l):
                  p_val = indep_test_func(data_matrix, i, j, set(k))

                  if self.max_tests_done:
                    return (g_old, sep_set_old)

                  if p_val > alpha:
                      if g.has_edge(i, j):
                          g.remove_edge(i, j)
                      sep_set[i][j] |= set(k)
                      sep_set[j][i] |= set(k)
                      break
              cont = True
      l += 1
      if cont is False:
          break
      
    return (g, sep_set)