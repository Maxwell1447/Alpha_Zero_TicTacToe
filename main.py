from TicTacToeNNet import *
from TicTacToeGame import *
from TicTacToeLogic import *
from TicTacToePlayers import *
from NNet import *
from Coach import *
from MCTS import *
from Arena import *
from utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'numItersForTrainExamplesHistory': 20,
})

g = TicTacToeGame()
nn = NNetWrapper(g)
c = Coach(g, nn, args)

c.learn()
