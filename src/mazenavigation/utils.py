import numpy as np
from queue import Queue
from enum import Enum


class CardinalDirections(Enum):
    NORTH = 0
    EAST = 1
    WEST = 2
    SOUTH = 3


def get_directions(startPoint: tuple, endPoint: tuple):
    assert startPoint != endPoint
    if startPoint[0] == endPoint[0]:
        return CardinalDirections.EAST if startPoint[1] < endPoint[1] else CardinalDirections.WEST
    elif startPoint[1] == endPoint[1]:
        return CardinalDirections.SOUTH if startPoint[0] < endPoint[0] else CardinalDirections.NORTH
    else:
        raise Exception("Not supported yet")


def find_shortest_path(maze: np.ndarray, start: tuple, end: tuple):

    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0]+dx, node[1]+dy)
            if (next_node == end):
                return path + [next_node]
            if (next_node[0] >= 0 and next_node[1] >= 0 and
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and
                    maze[next_node] < 1 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


def remove_cycles_in_path(path: list[tuple]):
    unique_path = []
    visited = set()
    for point in path:
        if point not in visited:
            unique_path.append(point)
            visited.add(point)

    return unique_path


def get_adjacent_indices(matrix: np.ndarray, point: tuple, condiction=None):
    rows, cols = matrix.shape
    row, col = point

    adjacent_indices = []
    for i in range(row-1, row+2):
        for j in range(col-1, col+2):
            if 0 <= i < rows and 0 <= j < cols:
                if condiction and condiction(matrix[i, j]):
                    adjacent_indices.append((i, j))
                elif not condiction:
                    adjacent_indices.append((i, j))

    return adjacent_indices


def get_indices_around_point(matrix: np.ndarray, point: tuple, condiction=None, radius=1, remove_itself=True):
    indices = set()
    rows, cols = matrix.shape
    row, col = point

    for i in range(max(0, row - radius), min(rows, row + radius + 1)):
        if abs(i - row) <= radius:
            if condiction and condiction(matrix[i, col]):
                indices.add((i, col))
            elif not condiction:
                indices.add((i, col))

    for j in range(max(0, col - radius), min(cols, col + radius + 1)):
        if abs(j - col) <= radius:
            if condiction and condiction(matrix[row, j]):
                indices.add((row, j))
            elif not condiction:
                indices.add((row, j))

    l = list(indices)
    if point in l:
        l.remove(point)
    return l


def convert_str_to_enum(type, obj: str):
    if isinstance(obj, type):
        return obj
    else:
        return type[obj]


def convert_dict_to_object(type, obj: dict):
    if isinstance(obj, type):
        return obj
    else:
        arg_names = list(type.__init__.__code__.co_varnames)
        arg_names.remove("self")
        arg = {x: obj[x] for x in arg_names}
        return type(**arg)
