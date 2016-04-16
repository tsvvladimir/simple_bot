import interface as bbox
import numpy as np

from pybrain.rl.environments.environment import Environment

class GameEnvironment(Environment):

    def __init__(self):
        self.finish_flag = True

    def getSensors(self):
        state = bbox.get_state()
        print 'state', state
        return state

    def performAction(self, action):
        print 'action', action
        self.finish_flag = bbox.do_action(action)