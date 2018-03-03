import numpy as np

test = np.ones((3,3))

check = np.zeros((3,3))

for i in range(3):
      sum_row = test[i,:].sum()

sum_row = test[1,:].sum()

for i in range(3):
      for j in range(3):
      
            
import turtle 
def drawSquare(my_turtle):
	my_turtle.color("red")
	for i in range(4):
		my_turtle.forward(100)
		my_turtle.right(90)

def drawSomethingCool(my_turtle):
	my_turtle.color("yellow")
	for i in range(100):
		my_turtle.forward(2 * 2 * i)
		my_turtle.right(90)
		my_turtle.right(45)

def drawSomethingCooler(my_turtle):
	my_turtle.color("blue")
	for i in range (36):
		drawSquare(my_turtle)
		my_turtle.right(10)

# create the turle's play pen
canvas = turtle.Screen()
canvas.bgcolor("black")

# create a turtle named mike
mike = turtle.Turtle()
mike.shapesize(2, 2, 2)
mike.color("yellow")
mike.shape("turtle")
mike.speed(10)

# drawSquare(mike)
# drawSomethingCool(mike)
drawSomethingCooler(mike)

# exits the screen when clicked
canvas.exitonclick()

def draw():
    # initialize an empty board
    board = ""

    # there are 5 rows in a standard tic-tac-toe board
    for i in range(5):
        # switch between printing vertical and horizontal bars
        if i%2 == 0:
            board += "|    " * 4
        else:
            board += " --- " * 3
        # don't forget to start a new line after each row using "\n"
        board += "\n"

    print(board)

draw()

def draw_board():
      board = ""
      
      for i in range(3):
            for j in range(3):
                  if test[i,j] == -1:
                        board += "x"
                  elif test[i,j] == 1:
                        board += "o"
                  else:
                        board += " "
                  if j < 2:
                        board += "|"
            if i <= 1:
                  board += "\n-----\n"
      
      print(board)
      
      
      
test = np.ones((3,3))
test.sum()


for state in s_prime:
      print(test_flip[state])














