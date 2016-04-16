import interface as bbox
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import Q
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from gameenvironment import GameEnvironment
from gametask import GameTask

def run_bbox(verbose=False):
    n_features = n_actions = max_time = -1

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()

    av_table = ActionValueTable(n_features, n_actions)
    av_table.initialize(0.2)
    print av_table._params
    learner = Q(0.5, 0.1)
    learner._setExplorer(EpsilonGreedyExplorer(0.4))
    agent = LearningAgent(av_table, learner)
    environment = GameEnvironment()
    task = GameTask(environment)
    experiment = Experiment(task, agent)

    while environment.finish_flag:
        experiment.doInteractions(1)
        agent.learn()
 
    bbox.finish(verbose=1)

 
if __name__ == "__main__":
    run_bbox(verbose=0)