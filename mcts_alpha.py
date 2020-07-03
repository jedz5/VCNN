# # -*- coding: utf-8 -*-
# """
# Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
# network to guide the tree search and evaluate the leaf nodes
#
# @author: Junxiao Song
# """
#
# import numpy as np
# import copy
# import policy_value_net_tensorflow
# import logging
# from H3_battle import logger
# # def rollout_policy_fn(board):
# #     """a coarse, fast version of policy_fn used in the rollout phase."""
# #     # rollout randomly
# #     action_probs = np.random.rand(len(board.availables))
# #     return zip(board.availables, action_probs)
#
#
# # def policy_value_fn(battle):
# #     """a function that takes in a state and outputs a list of (action, probability)
# #     tuples and a score for the state"""
# #     # return uniform probabilities and 0 score for pure MCTS
# #     legals = battle.curStack.legalMoves()
# #     action_probs = np.ones(len(legals))/len(legals)
# #     return zip(legals, action_probs), 0
#
# def softmax(x):
#     probs = np.exp(x - np.max(x))
#     probs /= np.sum(probs)
#     return probs
#
# class StateNode(object):
#     def __init__(self,parent,side,left_base,right_base,name=0):
#         self._parent = parent
#         self.side = side
#         self._actions = {}  # a map from action to TreeNode
#         self._n_visits = 0
#         self._Q = 0
#         self.left_base = left_base
#         self.right_base = right_base
#         self.name = name
#     def expand(self, action_priors,left_value,right_value):
#         """Expand tree by creating new children.
#         action_priors: a list of tuples of actions and their prior probability
#             according to the policy function.
#         """
#         self.left_value = left_value
#         self.right_value = right_value
#         for action, prob in action_priors:
#             if action not in self._actions:
#                 self._actions[action] = ActionNode(self, prob,self.side,self.name)
#
#     def select(self, c_puct):
#         """Select action among children that gives maximum action value Q
#         plus bonus u(P).
#         Return: A tuple of (action, next_node)
#         """
#         return max(self._actions.items(),
#                    key=lambda act_node: act_node[1].get_value(c_puct))
#     def update(self, left,right,value = -2):
#         """Update node values from leaf evaluation.
#         leaf_value: the value of subtree evaluation from the current player's
#             perspective.
#         """
#         # Count visit.
#         self._n_visits += 1
#         if value != -2:
#             leaf_value = value
#         else:
#             if self.side == 1:
#                 leaf_value = 1.0 - (left*self.left_value).sum()/((self.left_base*self.left_value).sum()+1e-10)
#             else:
#                 leaf_value = (left*self.left_value).sum()/((self.left_base*self.left_value).sum()+1e-10) - (right*self.right_value).sum()/((self.right_base*self.right_value).sum()+1e-10)
#
#             # Update Q, a running average of values for all visits.
#         self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
#     def update_recursive(self, left,right,valueL = -2,valueR = -2):
#         """Like a call to update(), but applied recursively for all ancestors.
#         """
#         # If it is not root, this node's parent should be updated first.
#         if(self.side == 1):
#             self.update(left,right,valueR)
#         else:
#             self.update(left, right, valueL)
#         if self._parent:
#             self._parent.update_recursive(left,right,valueL,valueR)
#
#     def is_leaf(self):
#         """Check if leaf node (i.e. no nodes below this have been expanded)."""
#         return self._actions == {}
#     def is_root(self):
#         return self._parent is None
# class ActionNode(object):
#     """A node in the MCTS tree.
#
#     Each node keeps track of its own value Q, prior probability P, and
#     its visit-count-adjusted prior score u.
#     """
#
#     def __init__(self, parent, prior_p,side,name=0):
#         self._parent = parent
#         self.side = side
#         self._states = {}  # states hash
#         self.curState = 0
#         self._n_visits = 0
#         self._Q = 0
#         self._u = 0
#         self._P = prior_p
#         self.name = name
#         self._aplayout = 0
#     def setCurentState(self,gameState):
#         stateHash = gameState.getHash()
#         left, leftBase, right, rightBase = gameState.getStackHPBySlots()
#         if stateHash not in self._states.keys():
#             self._states[stateHash] = StateNode(self, gameState.currentPlayer(), left, right, gameState.cur_stack.name)
#         else:
#             self._states[stateHash].left_base = left
#             self._states[stateHash].right_base = right
#             logger.debug('found same hash ')
#         self.curState = self._states[stateHash]
#     def update(self, left,right,value = -2):
#         """Update node values from leaf evaluation.
#         leaf_value: the value of subtree evaluation from the current player's
#             perspective.
#         """
#         # Count visit.
#         self._n_visits += 1
#         if value != -2:
#             leaf_value = value
#             logger.info('{} side {}, q={}, n={},p_n={},p={},update leaf_value = {}'.format(self.name, self.side,self._Q,self._n_visits,self._parent._n_visits,self._P,leaf_value))
#         else:
#             if self.side == 1:
#                 leaf_value = 1.0 - (left * self._parent.left_value).sum() / ((self._parent.left_base * self._parent.left_value).sum()+1e-10)
#             else:
#                 leaf_value = (left * self._parent.left_value).sum() / ((self._parent.left_base * self._parent.left_value).sum()+1e-10) - (right * self._parent.right_value).sum() / ((self._parent.right_base * self._parent.right_value).sum()+1e-10)
#             logger.info('{} side {}, q={}, n={},p_n={},p={},update from simulate leaf_value = {}'.format(self.name, self.side, self._Q,
#                                                                                     self._n_visits,self._parent._n_visits, self._P,
#                                                                                     leaf_value))
#
#         self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
#
#     def update_recursive(self, left,right,valueL = -2,valueR = -2):
#         """Like a call to update(), but applied recursively for all ancestors.
#         """
#         # If it is not root, this node's parent should be updated first.
#         if(self.side == 1):
#             self.update(left,right,valueR)
#         else:
#             self.update(left, right, valueL)
#         if self._parent:
#             self._parent.update_recursive(left,right,valueL,valueR)
#
#     def get_value(self, c_puct):
#         """Calculate and return the value for this node.
#         It is a combination of leaf evaluations Q, and this node's prior
#         adjusted for its visit count, u.
#         c_puct: a number in (0, inf) controlling the relative impact of
#             value Q, and prior probability P, on this node's score.
#         """
#         self._u = (c_puct * self._P *
#                    np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
#         return self._Q + self._u
#
#
#
#
# class MCTS(object):
#     """An implementation of Monte Carlo Tree Search."""
#
#     def __init__(self, policy_value_fn, c_puct=5, n_playout=10000,battle = 0):
#         """
#         policy_value_fn: a function that takes in a board state and outputs
#             a list of (action, probability) tuples and also a score in [-1, 1]
#             (i.e. the expected value of the end game score from the current
#             player's perspective) for the current player.
#         c_puct: a number in (0, inf) that controls how quickly exploration
#             converges to the maximum-value policy. A higher value means
#             relying on the prior more.
#         """
#         side = battle.currentPlayer()
#         left, leftBase, right, rightBase = battle.getStackHPBySlots()
#         self._root = StateNode(None,side,left,right,battle.curStack.name)
#         self._policy = policy_value_fn
#         self._c_puct = c_puct
#         self._n_playout = n_playout
#         self.side = side
#
#     def _playout(self, state):
#         """Run a single playout from the root to the leaf, getting a value at
#         the leaf and propagating it back through its parents.
#         State is modified in-place, so a copy must be provided.
#         """
#         stateNode = self._root
#         level = 0
#         while(1):
#             if stateNode.is_leaf():
#                 break
#             level += 1
#             # Greedily select next move.
#             action_id, actionNode = stateNode.select(self._c_puct)
#             act = state.indexToAction(action_id)
#             if level == 1:
#                 logger.info("{}.{} playout {} action {}_start".format(state.batId, level, state.cur_stack.name,
#                                                                       state.action2Str(action_id)))
#             else:
#                 logger.info("{}.{} playout {} action {}".format(state.batId, level, state.cur_stack.name, state.action2Str(action_id)))
#             state.doAction(act)
#             state.checkNewRound(1)
#             #actionNode._aplayout = "{}.{}".format(state.batId,level)
#             #actionNode._aL = actionNode._parent.left_base
#             #actionNode._aR = actionNode._parent.right_base
#             actionNode.setCurentState(state)
#             stateNode = actionNode.curState
#         # Evaluate the leaf using a network which outputs a list of
#         # (action, probability) tuples p and also a score v in [-1, 1]
#         # for the current player.
#         action_probs, leaf_valueL,leaf_valueR,fvalue_left,fvalue_right = self._policy(state)
#         # Check for end of game.
#         end,winner = state.end()
#         if not end:
#             logger.info("playout not end, update")
#             stateNode.expand(action_probs,fvalue_left,fvalue_right)
#             stateNode.update_recursive(0,0,leaf_valueL,leaf_valueR)
#         else:
#             logger.info("playout end, winner {}".format(winner))
#             # for end state，return the "true" leaf_value
#             stateNode.left_value = fvalue_left
#             stateNode.right_value = fvalue_right
#             # if winner == -1:  # tie
#             #     leaf_value = 0.0
#             #     stateNode.update_recursive(0,0,leaf_valueL,leaf_valueR)
#             # else:
#             left, leftBase, right, rightBase = state.getStackHPBySlots()
#             stateNode.update_recursive(left,right)
#
#         # Update value and visit count of nodes in this traversal.
#
#
#     def get_move_probs(self, battle, temp=1e-3):
#         """Run all playouts sequentially and return the available actions and
#         their corresponding probabilities.
#         state: the current game state
#         temp: temperature parameter in (0, 1] controls the level of exploration
#         """
#         for n in range(self._n_playout):
#             state_copy = battle.getCopy()
#             self._playout(state_copy)
#             #logger.info(state_copy.path)
#
#         # calc the move probabilities based on visit counts at the root node
#         act_visits = [(act, node._n_visits)
#                       for act, node in self._root._actions.items()]
#         acts, visits = zip(*act_visits)
#         act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
#
#         return acts, act_probs
#
#     def update_with_move(self, last_move,battle):
#         """Step forward in the tree, keeping everything we already know
#         about the subtree.
#         """
#         if last_move in self._root._actions:
#             actionNode= self._root._actions[last_move]
#             actionNode.setCurentState(battle)
#             stateNode = actionNode.curState
#             self._root = stateNode
#             self._root._parent = None
#         else:
#             left, leftBase, right, rightBase = battle.getStackHPBySlots()
#             self._root = StateNode(None, battle.currentPlayer(), left, right, battle.cur_stack.name)
#
#     def __str__(self):
#         return "MCTS"
#
#
# class MCTSPlayer(object):
#     """AI player based on MCTS"""
#
#     def __init__(self, policy_value_function,
#                  c_puct=5, n_playout=2000, is_selfplay=0,battle = 0):
#         side = battle.currentPlayer()
#         self.mcts = MCTS(policy_value_function, c_puct, n_playout,battle)
#         self._is_selfplay = is_selfplay
#         self.player = side
#
#     def set_player_ind(self, p):
#         self.player = p
#
#     def reset_player(self,battle):
#         self.mcts.update_with_move(-1,battle)
#
#     def getAction(self, battle, temp=1e-3, return_prob=1):
#         sensible_moves = battle.cur_stack.legalMoves()
#         move_probs = np.zeros(battle.bTotalFieldSize)
#         if len(sensible_moves) > 0:
#             acts, probs = self.mcts.get_move_probs(battle, temp)
#             move_probs[list(acts)] = probs
#             if self._is_selfplay:
#                 # add Dirichlet Noise for exploration (needed for
#                 # self-play training)
#                 move = np.random.choice(
#                     acts,
#                     p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
#                 )
#             else:
#                 # with the default temp=1e-3, it is almost equivalent
#                 # to choosing the move with the highest prob
#                 move = np.random.choice(acts, p=probs)
#                 # reset the root node
#                 self.mcts.update_with_move(-1,battle)
# #                location = board.move_to_location(move)
# #                logger.info("AI move: %d,%d\n" % (location[0], location[1]))
#
#             if return_prob:
#                 return move, move_probs
#             else:
#                 return move
#         else:
#             logger.info("WARNING: the board is full")
#
#     def __str__(self):
#         return "MCTS {}".format(self.player)
