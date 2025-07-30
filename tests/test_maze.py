import unittest
import os
import numpy as np

from mazenavigation import *

import mazenavigation as maznav


class TestMaze(unittest.TestCase):

    def test_load(self):
        folder = "./tests/test"
        os.makedirs(folder, exist_ok=True)

        maze = Maze()
        maze.create(10, weight_strategy=UniformWeightsAssignament(0.8))
        maze.save(os.path.join(folder, "test.json"))
        self.assertTrue(maze.is_loaded)
        maze.clear()
        self.assertFalse(maze.is_loaded)
        maze.load(os.path.join(folder, "test.json"))
        self.assertTrue(maze.is_loaded)

    def test_draw(self):
        folder = "./tests/test"
        os.makedirs(folder, exist_ok=True)

        maze = Maze()
        for i, d in enumerate(np.arange(0, 1, 0.1)):
            maze.create(20, density_factor=d,  weight_strategy=UniformWeightsAssignament(0.5),
                        seed=1234, random_points=True)
            best_path = maze.find_shortest_path()
            maze.draw(os.path.join(folder, f"./test_draw_{i}.svg"), best_path)

    def test_weights(self):
        folder = "./tests/test"
        os.makedirs(folder, exist_ok=True)

        dim = 40
        density = 0
        noise = 0.2
        factor = 0.5
        rdnpoint = True
        seed = None

        maze = Maze()
        maze.create(
            dim=dim,
            density_factor=density,
            weight_strategy=UniformWeightsAssignament(factor),
            seed=seed, random_points=rdnpoint)
        maze.draw(os.path.join(folder, "weight_uniform"))
        maze.plot_weights_distribution(os.path.join(folder, "dis_weight_uniform"))

        maze.create(
            dim=dim,
            density_factor=density,
            weight_strategy=GaussianWeightsAssignament(0.5, 0.1, factor),
            seed=seed, random_points=rdnpoint)
        maze.draw(os.path.join(folder, "weight_guassian"))
        maze.plot_weights_distribution(os.path.join(folder, "dis_weight_guassian"))

        maze.create(
            dim=dim,
            density_factor=density,
            weight_strategy=BinomialWeightsAssignament(10, 0.5, factor),
            seed=seed, random_points=rdnpoint)
        maze.draw(os.path.join(folder, "weight_binomial"))
        maze.plot_weights_distribution(os.path.join(folder, "dis_weight_binomial"))

        maze.create(
            dim=dim,
            density_factor=density,
            weight_strategy=StaticWeightsAssignament(0.2),
            seed=seed, random_points=rdnpoint)
        maze.draw(os.path.join(folder, "weight_static"))
        maze.plot_weights_distribution(os.path.join(folder, "dis_weight_static"))


        for v in np.arange(0.1, 1.1, 0.1):
            v = np.round(v, 2)
            name = str(v).replace(".", "_")
            maze.create(
                dim=dim,
                density_factor=density,
                weight_strategy=EdgeWeightsAssignament(v, noise, factor),
                seed=seed, random_points=rdnpoint)
            maze.draw(os.path.join(folder, f"weight_edge_{name}"))
            maze.plot_weights_distribution(os.path.join(folder, f"dis_weight_edge_{name}"))

            maze.create(
                dim=dim,
                density_factor=density,
                weight_strategy=CenterWeightsAssignament(v, noise, factor),
                seed=seed, random_points=rdnpoint)
            maze.draw(os.path.join(folder, f"weight_center_{name}"))
            maze.plot_weights_distribution(os.path.join(folder, f"dis_weight_center_{name}"))

            maze.create(
                dim=dim,
                density_factor=density,
                weight_strategy=DiagonalWeightsAssignament(v, noise, factor),
                seed=seed, random_points=rdnpoint)
            maze.draw(os.path.join(folder, f"weight_diagonal_{name}"))
            maze.plot_weights_distribution(os.path.join(folder, f"dis_weight_diagonal_{name}"))

            maze.create(
                dim=dim,
                density_factor=density,
                weight_strategy=PathWeightsAssignament(v,noise, factor),
                seed=seed, random_points=rdnpoint)
            maze.draw(os.path.join(folder, f"weight_path_{name}"))
            maze.plot_weights_distribution(os.path.join(folder, f"dis_weight_path_{name}"))

    def test_explore(self):
        folder = "./tests/test"
        os.makedirs(folder, exist_ok=True)

        agent = RandomAgent()

        maze = Maze()
        maze.create(dim=40,
                    density_factor=0.1,
                    weight_strategy=UniformWeightsAssignament(0.5),
                    seed=1234,
                    random_points=True)

        path, cost, time = maze.explore(
            agent, stop_criteria=lambda delta, m: m < 10)
        maze.draw(os.path.join(folder, f"./test_explore.svg"),
                  path=path)
        
