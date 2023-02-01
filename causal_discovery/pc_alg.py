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
from causal_discovery.causal_discovery_base import CausalDiscoveryBase


class PCAlgorithm(CausalDiscoveryBase):
  """An implementation of the PC algorithm adapted from [1].

  [1] https://github.com/keiichishima/pcalg/blob/master/pcalg.py.
  """

  def __init__(self, treatment_node="X", outcome_node="Y", alpha=0.05, 
               use_ci_oracle=True, graph_true=None, enable_logging=False,
               max_tests=None):
    super(PCAlgorithm, self).__init__(treatment_node=treatment_node,
                                      outcome_node=outcome_node, alpha=alpha,
                                      use_ci_oracle=use_ci_oracle,
                                      graph_true=graph_true,
                                      enable_logging=enable_logging,
                                      max_tests=max_tests)
  
  def _estimate_skeleton(self, indep_test_func, data_matrix, alpha, **kwargs):
    nx = networkx
    np = numpy

    def is_ci_test_needed(g, i, j, cond):
      if frozenset([i, j, frozenset(cond)]) in tests_done:
        return False
        
      return True

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    tests_done = set()

    g = self._create_complete_graph(node_ids)

    l = 0
    while True:
      cont = False
      node_ids_shuffled = copy.copy(list(node_ids))
      np.random.shuffle(node_ids_shuffled)
      for (i, j) in permutations(node_ids, 2):
        adj_i = list(g.neighbors(i))
        if j not in adj_i:
          continue
        else:
          adj_i.remove(j)
        if len(adj_i) >= l:
          adj_i_sorted = sorted(adj_i, key=lambda n: 0 if n == self.tmt_id else 1)
          for k in combinations(adj_i_sorted, l):
              if not is_ci_test_needed(g, i, j, set(k)):
                continue

              p_val = indep_test_func(data_matrix, i, j, set(k))
              tests_done |= {frozenset([i, j, frozenset(k)])}
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

  def _estimate_cpdag(self, skel_graph, sep_set, tmt=None):
    dag = skel_graph.to_directed()
    dag = orient_colliders(dag, sep_set, tmt=tmt)
    return apply_meek_rules(dag)

  def run(self, data):
    pd = pandas
    nx = networkx

    cols = data.columns
    self.cols = cols
    tmt_id = list(cols).index(self.tmt_node)
    self.tmt_id = tmt_id
    sep_set_dict = collections.defaultdict(dict)
    skeleton, sep_set = self._estimate_skeleton(self._partial_corr_independence_test,
                                                pd.DataFrame(data=data.values,
                                                             columns=range(data.values.shape[1])),
                                                alpha=self.alpha)
    cpdag = self._estimate_cpdag(skeleton, sep_set, tmt=tmt_id)
    cpdag = nx.relabel_nodes(cpdag, {i: cols[i] for i in range(len(cols))})
    skeleton = nx.relabel_nodes(skeleton, {i: cols[i] for i in range(len(cols))})


    def node_ids_to_labels(id_set):
      return set(cols[i] for i in id_set)

    for i, sep_val in enumerate(sep_set):
      for j, id_set in enumerate(sep_val):
        sep_set_dict[cols[i]][cols[j]] = node_ids_to_labels(id_set)

    return cpdag, skeleton, sep_set_dict
