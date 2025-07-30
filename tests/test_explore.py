import unittest
import os
import numpy as np

from mazenavigation import *


class TestExplore(unittest.TestCase):
    def test_explore(self):
        folder = "./tests/test"
        os.makedirs(folder, exist_ok=True)

        agent = RandomAgent()

        maze = Maze()
        maze.create(dim=10,
                    density_factor=0.1,
                    weight_strategy=UniformWeightsAssignament(0.5),
                    seed=1234,
                    random_points=True)

        exp = Exploration(
            maze=maze,
            stop_criteria=ThresholdStoppingCriteria(max_len_path=20),
            use_register=True
        )
        result = exp.single(
            agent=agent
        )
        maze.draw(os.path.join(folder, f"./test_explore.svg"),
                  path=result.path)
