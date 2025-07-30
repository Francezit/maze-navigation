import numpy as np
import os
import json
import shutil

from .utils import *
from .weightsassignament import WeightsAssignament

MAZE_WALL_CODE = 1


class Maze():
    __weighted_maze: np.ndarray
    __enter_maze: tuple
    __exit_maze: tuple

    def __init__(self):
        self.clear()

    @property
    def entrance(self):
        """Alias per enter_point per compatibilità"""
        return self.enter_point

    @property 
    def exit(self):
        """Alias per exit_point per compatibilità"""
        return self.exit_point

    @property
    def is_loaded(self):
        return self.__weighted_maze is not None

    @property
    def size(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return self.__weighted_maze.size

    @property
    def shape(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return self.__weighted_maze.shape

    @property
    def enter_point(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return self.__enter_maze

    @property
    def exit_point(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return self.__exit_maze

    @property
    def number_cells(self):
        assert self.is_loaded, "Maze is not loaded yet"
        num_walls = np.sum(self.__weighted_maze == MAZE_WALL_CODE)
        return self.__weighted_maze.size-num_walls

    @property
    def number_walls(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return np.sum(self.__weighted_maze == MAZE_WALL_CODE)

    @property
    def weights(self):
        assert self.is_loaded, "Maze is not loaded yet"
        w = self.__weighted_maze.reshape((self.__weighted_maze.size,))
        w = np.delete(w, np.where(w == MAZE_WALL_CODE)[0])
        return w

    @property
    def avg_weights(self):
        return np.nanmean(self.weights)

    @property
    def std_weights(self):
        return np.nanstd(self.weights)

    @property
    def total_weights(self):
        return np.nansum(self.weights)

    @property
    def cell_density(self):
        n_cell = self.number_cells
        return (n_cell)/self.__weighted_maze.size

    @property
    def wall_density(self):
        n_wall = self.number_walls
        return (n_wall)/self.__weighted_maze.size

    @property
    def matrix(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return self.__weighted_maze

    def clear(self):
        self.__weighted_maze = None
        self.__enter_maze = None
        self.__exit_maze = None

    def load(self, filename):
        with open(filename, "r") as fp:
            data = json.load(fp)
            self.__weighted_maze = np.array(data["weighted_maze"])
            self.__enter_maze = tuple(data["enter_maze"])
            self.__exit_maze = tuple(data["exit_maze"])

    def save(self, filename):
        assert self.is_loaded, "Maze is not loaded yet"
        with open(filename, "w") as fp:
            data = {
                "weighted_maze": self.__weighted_maze.tolist(),
                "enter_maze": self.__enter_maze,
                "exit_maze": self.__exit_maze
            }
            json.dump(data, fp)
    
    def create(self, dim: int, weight_strategy: WeightsAssignament = None, density_factor: float = 0,
               random_points: bool = False, seed: int = None):  

        rdn = np.random.RandomState(seed)

        # Create a grid filled with walls
        maze = np.ones((dim*2+1, dim*2+1))*MAZE_WALL_CODE

        # Define the starting point
        x, y = (0, 0)
        maze[2*x+1, 2*y+1] = 0

        # Initialize the stack with the starting point
        # https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            rdn.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                    maze[2*nx+1, 2*ny+1] = 0
                    maze[2*x+1+dx, 2*y+1+dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        if density_factor and density_factor > 0:
            n = int(np.floor(maze.size*density_factor))
            for _ in range(n):
                r = rdn.randint(1, maze.shape[0]-1)
                c = rdn.randint(1, maze.shape[1]-1)
                maze[r, c] = 0

        # compute the entrance and the exit
        enter_point: tuple
        exit_point: tuple
        if random_points:
            y = rdn.randint(0, maze.shape[0])
            x = rdn.randint(0, maze.shape[1])

            if rdn.rand() < 0.5:
                while maze[1, x] == MAZE_WALL_CODE:
                    x = (x+1) % maze.shape[1]
                enter_point = (0, x)
            else:
                while maze[maze.shape[0]-2, x] == MAZE_WALL_CODE:
                    x = (x+1) % maze.shape[1]
                enter_point = (maze.shape[0]-1, x)

            if rdn.rand() < 0.5:
                while maze[y, 1] == MAZE_WALL_CODE:
                    y = (y+1) % maze.shape[0]
                exit_point = (y, 0)
            else:
                while maze[y, maze.shape[1]-2] == MAZE_WALL_CODE:
                    y = (y+1) % maze.shape[0]
                exit_point = (y, maze.shape[1]-1)

            if rdn.rand() < 0.5:
                t = exit_point
                exit_point = enter_point
                enter_point = t
        else:
            enter_point = (1, 0)
            exit_point = (maze.shape[0]-2, maze.shape[1]-1)

        # set the distributions of weights
        if weight_strategy:
            w = weight_strategy(
                maze_shape=maze.shape, rdn=rdn,
                enter=enter_point, exit=exit_point
            )
            w = np.round(w, 7)
            assert np.min(w) >= 0 and np.max(w) < MAZE_WALL_CODE

            mask = maze == 0
            maze[mask] = w[mask]

        maze[enter_point] = np.nan
        maze[exit_point] = 0

        self.__weighted_maze = maze
        self.__enter_maze = enter_point
        self.__exit_maze = exit_point
    
    def animation(self, filename: str, path: list, duration: int = 100, frame_numbers: int = -1, temp_folder: str = None):
        if frame_numbers is None or frame_numbers < 0:
            precision: int = 1
        elif frame_numbers > 0:
            precision: int = int(np.ceil(len(path)/frame_numbers))
        else:
            raise Exception("frame_numbers not valid")

        if temp_folder is None:
            temp_folder = f"./images_gif_{np.random.randint(1000,10000)}"

        import contextlib
        import shutil
        from PIL import Image
        
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)


        try:
            # compute number of frames
            path_len = len(path)
            frame_indices = list(range(0, path_len, precision))
            if frame_indices[-1] != path_len-1:
                frame_indices.append(path_len-1)

            # generate frame
            sub_path = []
            frame_sequence = []
            for index in range(path_len):
                sub_path.append(path[index])
                if index in frame_indices:
                    f_temp = os.path.join(temp_folder, f"IMG_{index}.png")
                    self.draw(f_temp, sub_path, show_current_position=True, show_weights=True)
                    frame_sequence.append(f_temp)
            del sub_path

            with contextlib.ExitStack() as stack:
                imgs = (stack.enter_context(Image.open(f))
                        for f in frame_sequence)
                img = next(imgs)
                img.save(fp=filename, format='GIF', append_images=imgs,
                         save_all=True, duration=duration, loop=0)
        finally:
            shutil.rmtree(temp_folder)

    def draw(self, filename: str, path: list = None, multi_paths: list = None, names: list = None, show_current_position: bool = False,show_weights=False):
        assert self.is_loaded, "Maze is not loaded yet"
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        maze = self.__weighted_maze
        enter_maze = self.__enter_maze
        exit_maze = self.__exit_maze

        fig, ax = plt.subplots(figsize=(10, 10))

        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        if show_weights:
            ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
        else:
            ax.imshow(np.floor(maze), cmap=plt.cm.binary, interpolation='nearest')


        # Draw the solution path if it exists
        if path:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, color='red', linewidth=3)

        if multi_paths:
            cmap = plt.cm.get_cmap("tab20c")
            for i, p in enumerate(multi_paths):
                x_coords = [x[1] for x in p]
                y_coords = [y[0] for y in p]
                ax.plot(x_coords, y_coords, color=cmap(i), linewidth=3)

        if names:
            ax.legend(names)

        ax.set_xticks([])
        ax.set_yticks([])

        # draw current position
        if show_current_position:
            ax.add_patch(
                Circle(xy=(path[-1][1], path[-1][0]), radius=0.5, color="red"))

        # Draw endpoints
        ax.add_patch(
            Circle(xy=(enter_maze[1], enter_maze[0]), radius=0.5, color="lime"))

        ax.add_patch(
            Circle(xy=(exit_maze[1], exit_maze[0]), radius=0.5, color="orange"))
        ax.grid(True)

        fig.savefig(filename)
        plt.close(fig)


    def find_shortest_path(self):
        assert self.is_loaded, "Maze is not loaded yet"
        return find_shortest_path(self.__weighted_maze, self.__enter_maze, self.__exit_maze)

    def set_points(self, enter_point: tuple, exit_point: tuple):
        """Set custom enter and exit points for the maze."""
        assert self.is_loaded, "Maze is not loaded yet"
        
        assert 0 <= enter_point[0] < self.shape[0], "Enter point row out of bounds"
        assert 0 <= enter_point[1] < self.shape[1], "Enter point col out of bounds"
        assert 0 <= exit_point[0] < self.shape[0], "Exit point row out of bounds"
        assert 0 <= exit_point[1] < self.shape[1], "Exit point col out of bounds"
        
        assert self.matrix[enter_point] != MAZE_WALL_CODE, "Enter point cannot be a wall"
        assert self.matrix[exit_point] != MAZE_WALL_CODE, "Exit point cannot be a wall"

        if self.__enter_maze:
            self.matrix[self.__enter_maze] = 0
        if self.__exit_maze:
            self.matrix[self.__exit_maze] = 0
        
        self.matrix[enter_point] = np.nan
        self.matrix[exit_point] = 0
        
        self.__enter_maze = enter_point
        self.__exit_maze = exit_point

__all__ = ["Maze", "MAZE_WALL_CODE"]
