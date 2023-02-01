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

class MBbyMBAlgorithm(SequentialLocalCausalDiscoveryBase):
  """The MB-by-MB algorithm [1] is an instantiation of sequential local causal
  discovery where a combination of the IAMB [2] and (local) PC algorithms is 
  used to perform local structure learning at each sequential step.

  [1] Wang, C., Zhou, Y., Zhao, Q., & Geng, Z. (2014). Discovering and orienting the edges connected to a target variable in a DAG via a sequential local learning approach. Computational Statistics & Data Analysis.
  [2] Tsamardinos, I., Aliferis, C. F., Statnikov, A. R., & Statnikov, E. (2003, May). Algorithms for large scale Markov blanket discovery. In FLAIRS conference.
  """

  def __init__(self, treatment_node="X", outcome_node="Y", alpha=0.05, 
               use_ci_oracle=True, graph_true=None, enable_logging=False,
               max_tests=None):
    super(MBbyMBAlgorithm, self).__init__(treatment_node=treatment_node,
                                      outcome_node=outcome_node, alpha=alpha,
                                      use_ci_oracle=use_ci_oracle,
                                      graph_true=graph_true,
                                      enable_logging=enable_logging,
                                      max_tests=max_tests,
                                      local_structure_learning_func=self._find_local_structure)

  def _find_local_structure(self, g, sep_set, main_node_id, data_matrix):

    def is_independent(i, j, k, ret_p_val=False):
      p_val = self._partial_corr_independence_test(data_matrix, i, j, k)
      is_indep = p_val > self.alpha
      return is_indep
    
    def do_local_pc(g, sep_set, markov_blanket, main_node_id):
      g = g.copy()
      sep_set = copy.deepcopy(sep_set)
      tmt = main_node_id

      def get_sepset(node, neighbors_X, size):
        for possible_mns in combinations(neighbors_X, size):
          possible_mns = set(possible_mns)
          if is_independent(tmt, node, possible_mns):
            return possible_mns

      markov_blanket = set(markov_blanket)
      mb_copy = set(markov_blanket)

      if self.use_ci_oracle:
        corr = np.zeros((len(self.graph_true.nodes()),))
      else:
        corr = data_matrix.corr()[tmt]
      size = 0
      while len(mb_copy) > size:
        nodes_to_check = sorted(list(mb_copy), key=lambda n:(corr[n]))
        for n in nodes_to_check:
          sep = get_sepset(n, mb_copy - {n}, size)
          if sep is not None:
            mb_copy -= {n}
            if g.has_edge(n, tmt):
              g.remove_edge(n, tmt)
              sep_set[n][tmt] = sep
              sep_set[tmt][n] = sep

        size += 1

      return g, sep_set

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    
    if sep_set is None:
      sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    g = g if g is not None else self._create_complete_graph(node_ids)
    g = g.copy()
    
    node_ids = set(node_ids)
    sep_set = copy.deepcopy(sep_set)
    markov_blanket = set()
    init_neighbors = set()

    if self.use_ci_oracle:
      corr = np.zeros((len(self.graph_true.nodes()),))
    else:
      corr = data_matrix.corr()[main_node_id]

    # forward pass.
    cont = True
    while cont:
      cont = False
      mb_copy = set(markov_blanket)
      nodes_to_check = (node_ids - {main_node_id} - mb_copy)
      nodes_to_check = sorted(list(nodes_to_check), key=lambda n:-(corr[n]))
      for n in nodes_to_check:
        if not is_independent(n, main_node_id, markov_blanket - {n}):
          markov_blanket |= {n}
          cont = True
        elif g.has_edge(n, main_node_id):
          g.remove_edge(n, main_node_id)
          sep_set[n][main_node_id] = markov_blanket - {n}
          sep_set[main_node_id][n] = markov_blanket - {n}
    
    # backward pass.
    mb_copy = set(markov_blanket)
    for n in mb_copy:
      if is_independent(n, main_node_id, markov_blanket - {n}):
        if g.has_edge(n, main_node_id):
          g.remove_edge(n, main_node_id)
          sep_set[n][main_node_id] = markov_blanket - {n}
          sep_set[main_node_id][n] = markov_blanket - {n}
        
        markov_blanket -= {n}
    
    g, sep_set = do_local_pc(g, sep_set, markov_blanket, main_node_id)
    return g, sep_set