import interface as bbox

from pybrain.rl.environments.task import Task

class GameTask(Task):
    def __init__(self, environment):
        self.env = environment
        self.lastreward = 0

    def getObservation(self):
        return self.env.getSensors()

    def performAction(self, action):
        self.env.performAction(action)

    def getReward(self):
        cur_reward = self.lastreward
        self.lastreward = bbox.get_score()
        print 'lastreward', self.lastreward
        return cur_reward

    @property
    def indim(self):
        return self.env.indim

    @property
    def outdim(self):
        return self.env.outdim