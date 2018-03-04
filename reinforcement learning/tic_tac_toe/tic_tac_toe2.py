from __future__ import print_function, division
from builtins import range, input

import numpy as np
import matplotlib.pyplot as plt

# board lengt
LENGTH = 3

class Agent:
   def __init__(self, eps=0.1, alpha=0.5):
      self.eps = eps
      self.alpha = alpha
      self.verbose = False
      self.state_history =  []
      
   def set_symbol(self, sym):
      self.sym = sym
   
   def set_verbose(self, v):
      self.verbose = v
   
   def set_V(self, V):
      self.V = V
   
   def take_action(self, env):
      r = np.random.rand()
      next_move = None
      if r < self.eps:
         if self.verbose:
            print("Taking random action")
            
         possible_moves = []
         for i in range(LENGTH):
            for j in range(LENGTH):
               if env.is_empty(i,j):
                  possible_moves.append((i,j))
         next_move = possible_moves[np.random.choice(len(possible_moves))]
      else:
         pos2value = {} # for debugging
         best_value = - 1000
         next_move = None
         
         for i in range(LENGTH):
            for j in range(LENGTH):
               if env.is_empty(i,j):
                  env.board[i,j] = self.sym
                  state = env.get_state()
                  env.board[i,j] = 0
                  pos2value[(i,j)] = self.V[state]
                  if self.V[state] > best_value:
                     best_value = self.V[state]
                     next_move = (i,j)
      
         # if verbose, draw the board with the values
         if self.verbose:
            print("Taking a greedy action")
            for i in range(LENGTH):
               print("------------------")
               for j in range(LENGTH):
                  if env.is_empty(i, j):
                     # print the value of the value function
                     print(" %.2f|" % pos2value[(i,j)], end="")
                  else:
                     print("  ", end="")
                     if env.board[i,j] == env.x:
                        print("x  |", end="")
                     if env.board[i,j] == env.o:
                        print("o  |", end="")
                     else:
                        print("   |", end="")
               print("")
            print("------------------")
            
         env.board[next_move] = self.sym
                  
   def clear_state_history(self):
      self.state_history = []
   
   def update_state_history(self, state):
      # env.get_state() will be passed as state
      self.state_history.append(state)
      
   def update(self, env):
      # at the end of the game
      reward = env.reward(self.sym)
      target = reward
      for prev in reversed(self.state_history):
         value = self.V[prev] + self.alpha * (target - self.V[prev])
         self.V[prev] = value
         target = value
         
      self.clear_state_history()
         
                  
class Environment:
   def __init__(self):
      self.board = np.zeros((LENGTH,LENGTH))
      self.x = -1
      self.o = 1
      self.ended = False
      self.winner = None
      self.num_states = 3**(LENGTH*LENGTH)
   
   def is_empty(self, i, j):
      return self.board[i, j] == 0
   
   def game_over(self, force_recalculate=False):
      if not force_recalculate and self.ended:
         return self.ended
      """ checking all rows for winner """
      for i in range(LENGTH):
         for player in (self.x, self.o):
            if self.board[i,:].sum() == player * LENGTH:
               self.winner = player
               self.ended = True
               return True
      """ check columns for winner """
      for j in range(LENGTH):
         for player in (self.x, self.o):
            if self.board[:,j].sum() == player * LENGTH:
               self.winner = player
               self.ended = True
               return True
      """ check diagonals for winner """
      for player in (self.x, self.o):
         if self.board.trace() == player * LENGTH:
            self.winner = player
            self.ended = True
            return True
         if np.fliplr(self.board).trace() == player * LENGTH:
            self.winner = player
            self.ended = True
            return True
      """ check if there is a draw """
      if np.all(self.board):
         self.winner = None
         self.ended = True
         return True
      """ game is not over """
      self.winner = None
      return False
               
   def get_state(self):
      """ use ternary system to get the integer value of all possible states
      3^(3*3) = 19683, t in [0,1,2], n  is the nth power 
      DECIMAL = BASE^(N-1)*b_N-1 + ... + BASE^1*b_1 + BASE^0*b_0 """
      n = 0
      s = 0
      for i in range(LENGTH):
         for j in range(LENGTH):
            if self.board[i,j] == 0:
               t = 0
            elif self.board[i,j] == self.x:
               t = 1
            elif self.board[i,j] == self.o:
               t = 2
            s += (3**n) * t
            n += 1
      return s
      
   def draw_board(self):
      for i in range(LENGTH):
         print("-------------")
         for j in range(LENGTH):
           print("  ", end="")
           if self.board[i,j] == self.x:
             print("x ", end="")
           elif self.board[i,j] == self.o:
             print("o ", end="")
           else:
             print("  ", end="")
         print("")
      print("-------------")
   
   def reward(self, sym):
      if not self.game_over():
         return 0
      # return 1 if the winner is sym
      return 1 if self.winner == sym else 0

class Human:
   def __init__(self):
      pass
   
   def set_symbol(self, sym):
      self.sym = sym
      
   def take_action(self, env):
      while True:
         # break if we make a leagal ve
         move = input("Enter the coordinates i,j for your next move(i,j=0..2): ")
         i,j = move.split(',')
         i = int(i)
         j = int(j)
         if env.is_empty(i,j):
            env.board[i,j] = self.sym
            break
   
# draw = playernumber, if draw = 0 board is not drawn
def play_game(p1, p2, env, draw=False):
   current_player = None
   
   while not env.game_over():
      if current_player == p1:
         current_player = p2
      else:
         current_player = p1
      
      if draw:
         if draw == 1 and current_player == p1:
            env.draw_board()
         elif draw == 2 and current_player == p2:
            env.draw_board()
      
      current_player.take_action(env)
      
      state = env.get_state()
      p1.update_state_history(state)
      p2.update_state_history(state)
   
   if draw :
      env.draw_board()

   p1.update(env)
   p2.update(env)


def get_state_winner_ended_list(env):
   results = []
   for n in range(env.num_states):
      board_digits =  np.array(permutation_digits(n, 3), int) - 1
      
      for k in range(len(board_digits)):
         i = int(np.floor(k/3))
         j = k % 3
         env.board[i,j] = board_digits[k]
         
      state = env.get_state()
      ended = env.game_over(force_recalculate = True)
      winner = env.winner
      results.append((state, winner, ended))
      
   return results

def permutation_digits(n, b):
   # converts a positive number n to a ternary number with 9 positions
   digits = np.zeros(9, int)
   k = 0
   while n > 0:
      digits[k] = n % b
      n = n // b
      k += 1
   return digits

def initializeVx(env, state_winner_ended_triples):
   V = np.zeros(env.num_states)
   for state, winner, ended in state_winner_ended_triples:
      if ended:
         if winner == env.x:
            V[state] = 1
         else:
            V[state] = 0
      else:
         V[state] = 0.5
   return V
   
   
def initializeVo(env, state_winner_ended_triples):
   V = np.zeros(env.num_states)
   for state, winner, ended in state_winner_ended_triples:
      if ended:
         if winner == env.o:
            V[state] = 1
         else:
            V[state] = 0
      else:
         V[state] = 0.5
   return V

# %%

if __name__ == '__main__':
   p1 = Agent()
   p2 = Agent()
   
   env = Environment()
    
   state_winner_ended_triples = get_state_winner_ended_list(env)

   Vx = initializeVx(env, state_winner_ended_triples)
   p1.set_V(Vx)
   Vo = initializeVo(env, state_winner_ended_triples)
   p2.set_V(Vo)
   
   # give symbols
   p1.set_symbol(env.x)
   p2.set_symbol(env.o)
   
   T = 10000
   for t in range(T):
      if t % 200 == 0:
         print(t)
      play_game(p1, p2, Environment())
     
   human = Human()
   human.set_symbol(env.o)
   while True:
      p1.set_verbose(True)
      play_game(p1, human, Environment(), draw = 2)
      
      answer = input("Play again? [y/n]: ")
      if answer and answer.lower()[0] == 'n':
         break
   
      
   