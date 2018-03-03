import numpy as np
import matplotlib.pyplot as plt

# board length
LENGTH = 3

class Agent:
      def __init__(self, eps=0.1, alpha=0.5):
            self.eps = eps
            self.alpha = alpha
            self.verbose = False
            self.state_history = []

      def setV(self, V):
            self.V = V

      def set_symbol(self, sym):
            self.sym = sym
      
      def set_verbose(self, v):
            # if b True print more information and drawing board
            self.verbose = v
      
      def take_action(self, env):
          # choose an action based on epsilon-greedy strategy
          r = np.random.rand()
          best_state = None
          if r < self.eps:
            # take a random action
            if self.verbose:
              print("Taking a random action")
      
            possible_moves = []
            for i in range(LENGTH):
              for j in range(LENGTH):
                if env.is_empty(i, j):
                  possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
          else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            pos2value = {} # for debugging
            next_move = None
            best_value = -1
            for i in range(LENGTH):
              for j in range(LENGTH):
                if env.is_empty(i, j):
                  # what is the state if we made this move?
                  env.board[i,j] = self.sym
                  state = env.get_state()
                  env.board[i,j] = 0 # don't forget to change it back!
                  pos2value[(i,j)] = self.V[state]
                  if self.V[state] > best_value:
                    best_value = self.V[state]
                    best_state = state
                    next_move = (i, j)
                    
            # if verbose, draw the board w/ the values
            if self.verbose:
              print("Taking a greedy action")
              for i in range(LENGTH):
                print("------------------")
                for j in range(LENGTH):
                  if env.is_empty(i, j):
                    # print the value
                    print(" %.2f|" % pos2value[(i,j)], end="")
                  else:
                    print("  ", end="")
                    if env.board[i,j] == env.x:
                      print("x  |", end="")
                    elif env.board[i,j] == env.o:
                      print("o  |", end="")
                    else:
                      print("   |", end="")
                print("")
              print("------------------")
      
          # make the move
          env.board[next_move[0], next_move[1]] = self.sym
            

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
      
      # Example board
     # -------------
     # | x |   |   |
     # -------------
     # |   |   |   |
     # -------------
     # |   |   | o |
     # -------------
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

class Human:
  def __init__(self):
    pass

  def set_symbol(self, sym):
    self.sym = sym

  def take_action(self, env):
    while True:
      # break if we make a legal move
      move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
      i, j = move.split(',')
      i = int(i)
      j = int(j)
      if env.is_empty(i, j):
        env.board[i,j] = self.sym
        break

  def update(self, env):
    pass

  def update_state_history(self, s):
    pass


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
      
def get_state_winner_ended_list(env):
      results = []
      
      for n in range(env.num_states):
          board_digits = np.array(toDigits(n, 3)) - 1
          for k in range(len(board_digits)):
              i = int(np.floor(k/3))
              j = k % 3
              env.board[i,j] = board_digits[k]
              
          state = env.get_state()
          ended = env.game_over(force_recalculate = True)
          winner = env.winner
          
          results.append((state, winner, ended))
      
      return results

def initialV_x(env, state_winner_triples):
  # initialize state values as follows
  # if x wins, V(s) = 1
  # if x loses or draw, V(s) = 0
  # otherwise, V(s) = 0.5
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.x:
        v = 1
      else:
        v = 0
    else:
      v = 0.5
    V[state] = v
  return V

def initialV_o(env, state_winner_triples):
  # this is (almost) the opposite of initial V for player x
  # since everywhere where x wins (1), o loses (0)
  # but a draw is still 0 for o
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.o:
        v = 1
      else:
        v = 0
    else:
      v = 0.5
    V[state] = v
  return V

""" original toDegits
def toDigits(n, b):
    # Convert a positive number n to its digit representation in base b.
    digits = []
    while n > 0:
        digits.insert(0, n % b)
        n  = n // b
    return digits """

def toDigits(n, b):
    """Convert a positive number n to its digit representation in base b."""
    digits = np.zeros(9)
    k = 0
    while n > 0:
        digits[k] = n % b
        n  = n // b
        k += 1
    return digits
 
if __name__ == "__main__":
  # train the agent
  p1 = Agent()
  p2 = Agent()

  # set initial V for p1 and p2
  env = Environment()
  state_winner_triples = get_state_winner_ended_list(env)


  Vx = initialV_x(env, state_winner_triples)
  p1.setV(Vx)
  Vo = initialV_o(env, state_winner_triples)
  p2.setV(Vo)

  # give each player their symbol
  p1.set_symbol(env.x)
  p2.set_symbol(env.o)

  T = 10000
  for t in range(T):
    if t % 200 == 0:
      print(t)
    play_game(p1, p2, Environment())

  # play human vs. agent
  # do you think the agent learned to play the game well?
  human = Human()
  human.set_symbol(env.o)
  while True:
    p1.set_verbose(True)
    play_game(p1, human, Environment(), display_board=2)
    # I made the agent player 1 because I wanted to see if it would
    # select the center as its starting move. If you want the agent
    # to go second you can switch the human and AI.
    answer = input("Play again? [Y/n]: ")
    if answer and answer.lower()[0] == 'n':
      break