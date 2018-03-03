import numpy as np
import matplotlib.pyplot as plt

# board length
LENGTH = 3

class Agent:
      def __init__(self, eps=0.1, alpha=0.5):
            self.eps = eps
            self.alpha = alpha
            self.verbdose = False
            self.state_history = []

      def setV(self, V):
            self.V = V

      def set_symbol(self, sym):
            self.sym = sym
      
      def set_verbose(self, v):
            # if b True print more information and drawing board
            self.verbose = v
      
      def take_action(self, env):
            # choose action based on epsilon-greedy strategty
            r = np.random.rand()
            best_state = None
            if r < self.eps:
                  # do random action
                  if self.verbose:
                        print('Taking random action')
                  
                  possible_moves = []
                  for i in range(LENGTH):
                        for j in range(LENGTH): 
                              if env.is_empty(i, j):
                                    possible_moves.append((i,j))
                              idx = np.random.choice(len(possible_moves))
                              next_move = possible_moves[idx]
            else:
                  pos2value = {} # for debugging
                  next_move = None
                  best_value = -1
                  # possible moves was found alredy
                  for i in range(LENGTH):
                        for j in range(LENGTH):
                              if env.is_empty(i,j):
                                    # what is the state if we made this move
                                    env.board[i,j] = self.sym
                                    state = env.get_state()
                                    env.board[i,j] = 0
                                    pos2value[(i,j)] = self.V[state] # for debugging
                                    if self.V[state] > best_value:
                                          best_value = self.V[state]
                                          best_state = state
                                          next_move = (i, j)
            
            
            
            env.board[next_move] = self.sym
            

      def reset_history(self):
            self.state_history = []
            
      def update_state_history(self, state):
            # cannot be in take_aciton, because take_action only happens once every other
            # iteration for each player
            # state history needs to be updated every iteration
            # state = env.get_state() will be passed
            self.state_history.append(state)
            
      def update(self, env):
            # update the value function like V(s) = V(s) + learning_rate * (V(s') - V(s))
            reward = env.reward(self.sym)
            target = reward
            for prev in reversed(self.state_history):
                  value = self.V[prev] + self.alpha * (target - self.V[prev])
                  self.V[prev] = value
                  target = value
            self.reset_history
            

class Environment:
      def __init__(self):
            self.board = np.zeros((LENGTH,LENGTH))
            self.x = -1 # represents an x on the board, player 1
            self.o = 1 # represents an o on the board, player 2
            self.winner = None # will be -1 for player 1 or +1 for player 2 
            self.ended = False # will be True at the end of the game
            self.num_states = 3**(LENGTH*LENGTH)
      
      def is_empty(self, i, j):
            return self.board[i,j] == 0

      def game_over(self, force_recalculate=False):
            if not force_recalculate and self.ended:
                  return self.ended
            
            # checking rows
            for i in range(LENGTH):
                  sum_row = 0
                  sum_row = self.board[i,:].sum()
                  for player in (self.x, self.o):
                        if sum_row == player*LENGTH:
                              self.winner = player
                              self.ended = True
                              return True
                        
            # checking columns
            for j in range(LENGTH):
                  sum_col = 0
                  sum_col = self.board[:,j].sum()
                  for player in (self.x, self.o):
                        if sum_col == player*LENGTH:
                              self.winner = player
                              self.ended = True
                              return True
                        
            # checking diagonals
            sum_diag1 = 0
            sum_diag2 = 0
            sum_diag1 = self.board.trace()
            sum_diag2 = self.board[::-1].trace() # flipped matrix
            for player in (self.x, self.o):
                  if sum_diag1 == player*LENGTH or sum_diag2 == player*LENGTH:
                        self.winner = player
                        self.ended = True                     
                        return True
            
            # check if there is a draw, sum of board == - 1 in this case
            # because player 1 always starts the game and has 5 pieces on the board
            #if np.all((self.board == 0) == False)
            if self.board.sum() == - 1:
                  self.winner == None
                  self.ended = True
                  return True
            
            # game is not over
            self.winner = None
            return False
      
      def draw_board(self):
#            board_print = ""
#            for i in range(LENGTH):
#                  for j in range(LENGTH):
#                        if self.board[i,j] == self.x:
#                              board_print += "x"
#                        elif self.board[i,j] == self.o:
#                              board_print += "o"
#                        else:
#                              board_print += " "
#                        if j < LENGTH-1:
#                              board_print += "|"
#                  if i < LENGTH-1:
#                        board_print += "\n-----\n"
#            print(board_print)
      
      def get_state(self):
            h = 0
            k = 0 # keep track of fields placed?
            # returns the state - h the hash number
            for i in range(LENGTH):
                  for j in range(LENGTH):
                        if self.board[i,j] == 0:
                              v = 0
                        elif self.board[i,j] == self.x:
                              v = 1
                        elif self.board[i,j] == self.o:
                              v = 2
                        h += (3**k) * v
                        k += 1
            return h 
      
      def reward(self, sym):
            # no reward until game is over
            if not self.game_over():
                  return 0
            # if game is over
            # sym will be self.x or self.o
            return 1 if self.winner == sym else 0


def play_game(p1, p2, env, display_board=False):
      current_player = None
      
      # game loop 
      while not env.game_over():
            # player one always starts the game
            if current_player == p1:
                  current_player = p2
            else:
                  current_player = p1
            
            if display_board:
                  if display_board == 1 and current_player == p1:
                        env.draw_board()
                  if display_board == 2 and current_player == p2:
                        env.draw_board()                  
                  
            # current player takes action
            current_player.take_action(env)
                        
            # update states histories
            state = env.get_state()
            p1.update_state_history(state)
            p2.update_state_history(state)
      
      if display_board:
            env.draw_board()
      
      # value function update
      p1.update(env)
      p2.update(env)
      
def get_winner_state(env):
      results = []
      for i in range(LENGTH):
            for j in range(LENGTH):
                  for k in range(-1, 2):
                        env.board[i,j] = k
                        state = env.get_state()
                        winner = env.winner
                        ended = env.ended
                        results.append((state, winner, ended))
                        
      return results
                        
      

if __name__ == "__main__":
      # set variables and play the game
      
      p1 = Agent()
      p2 = Agent()
      
      env = Environment()
