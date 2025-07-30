import random
import numpy as np

from .utils import CardinalDirections, convert_str_to_enum


class DirectionInfo():

    @property
    def adjacent_cell(self):
        return self.cells[0]

    @property
    def adjacent_weight(self):
        return self.weights[0]

    @property
    def total_weight(self):
        return np.sum(self.weights)

    @property
    def n_cells(self):
        return len(self.cells)

    def get_visibile_weights(self, visibility: int=None):
        if visibility is None or np.isinf(visibility):
            return self.weights
        else:
            return self.weights[0:min(int(visibility), len(self.weights))]

    def get_visibile_cells(self, visibility: int, cell_corner: int = 0):
        c: list[tuple]
        if visibility is None or np.isinf(visibility):
            c = self.cells
        else:
            c = self.cells[0:min(int(visibility), len(self.cells))]

        return c

    def __init__(self, cells: list, weights: list, orientation: CardinalDirections) -> None:
        self.cells = cells
        self.weights = weights
        self.orientation = convert_str_to_enum(
            CardinalDirections,
            orientation
        )

    def __repr__(self) -> str:
        return f"{self.orientation.name}: {self.adjacent_cell} {self.total_weight} -> {self.n_cells} cells"

    def to_dict(self):
        return {
            "cells": self.cells,
            "weights": self.weights,
            "orientation": self.orientation.name,
            "n_cells": self.n_cells,
            "total_weight": self.total_weight,
            "adjacent_cell": self.adjacent_cell,
            "adjacent_weight": self.adjacent_weight
        }


class Agent:
    __name: str

    @property
    def class_type(self):
        raise Exception("Not implemented")

    @property
    def name(self) -> str:
        return self.__name

    def __init__(self, name: str = None) -> None:
        self.__name = name if name else f"A{random.randint(1000,9999)}"

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
        # current : contains the coordinate of the current cell
        # directions: contains all the directions. it is a list of DirectionInfo
        raise Exception("Not implemented")

    def get_settings(self) -> dict:
        raise Exception("Not implemented")
    
    def get_visibility(self) -> int:
        raise Exception("Not implemented")

    def to_dict(self) -> dict:
        info = self.get_settings()
        info["name"] = self.__name
        info["class_type"] = self.class_type
        return info

    def __repr__(self) -> str:
        s = self.get_settings()
        return f"{self.class_type}:{self.name} ({','.join([str(x) for x in s.values()])})"

    def __str__(self) -> str:
        s = self.get_settings()
        return f"{self.class_type}:{self.name} ({','.join([str(x) for x in s.values()])})"
    

class RandomAgent(Agent):
    def __init__(self, visibility: int = 0.5, **kargs) -> None:
        super().__init__(**kargs)

        self.__visibility = visibility
        assert self.__visibility > 0, "Visibility must be greater than 0"

    @property
    def class_type(self):
        return "Random"

    def get_settings(self) -> dict:
        return {}
    
    def get_visibility(self) -> int:
        return self.__visibility

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
       
        return random_state.randint(0, len(directions))


class StaticAgent(Agent):
    # dynamic parameters
    __alfa: float
    __beta: float
    __delta_1: float
    __delta_2: float

    # static parameters
    __visibility: int
    __memory_size: int

    # private variables
    __memory: list
     
    @property
    def class_type(self):
        return "Static"

    def __init__(self, alfa: float = 0.5, beta: float = 0.5, delta_1: float = 0.5, delta_2: float = 0.5,
                 visibility: int = 1, memory_size: int = 10, **kargs) -> None:
        super().__init__(**kargs)

        self.__alfa = alfa
        self.__beta = beta
        self.__delta_1 = delta_1
        self.__delta_2 = delta_2
        assert self.__delta_1+self.__delta_2 == 1, "delta_1 + delta_2 must be equalt to 1"

        self.__visibility = visibility
        self.__memory_size = memory_size

        self.__memory = []

    def get_settings(self) -> dict:
        base_dict = {}
        base_dict["alfa"] = self.__alfa
        base_dict["beta"] = self.__beta
        base_dict["delta_1"] = self.__delta_1
        base_dict["delta_2"] = self.__delta_2
        base_dict["visibility"] = self.__visibility
        base_dict["memory_size"] = self.__memory_size
        return base_dict

    def __compute_probabilities(self, scores: list[float], arg: float):

        sum_exp_w = np.sum(np.exp(scores))

        probs = np.empty(shape=(len(scores)))
        for index, score in enumerate(scores):
            p = np.exp(score)/sum_exp_w
            probs[index] = 1-arg+(2*arg-1)*p
        probs = probs/probs.sum()
        return probs

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:

        n_dirs = len(directions)

        # update memory
        if self.__memory_size > 0:
            self.__memory.append(current)
            if len(self.__memory) > self.__memory_size:
                del self.__memory[0]

        # if there are not other directions, the agent selects the only available direction
        if n_dirs == 1:
            return 0

        # compute the probabilities for each direction based on the cell weights
        weights = [
            np.sum(x.get_visibile_weights(self.__visibility))
            for x in directions
        ]
        cell_weights_probs = self.__compute_probabilities(weights, self.__alfa)

        # compute the probabilities for each direction based on its memory
        if self.__memory_size is None or self.__memory_size > 0:
            path_scores = []
            current_path: list = self.__memory.copy()
            for direction in directions:
                next_cells = direction.get_visibile_cells(self.__visibility)

                # TODO Ottimizare
                count = 0
                for i in next_cells:
                    for j in current_path:
                        if i == j:
                            count += 1
                path_scores.append(count)
            path_probs = self.__compute_probabilities(path_scores, self.__beta)
        else:
            path_probs = np.ones(shape=(n_dirs,))/n_dirs

        # compute the final probabilities
        probs = (self.__delta_1 * cell_weights_probs) + \
            (self.__delta_2 * path_probs)

        # select one direction
        val = random_state.choice(list(range(n_dirs)), p=probs)
        return int(val)
    
    #Aggiunto il set beta
    def set_beta(self, new_beta: float):
        self.__beta = new_beta  # Aggiorna l'attributo privato __beta

    #Aggiunto il get visibility
    @property
    def get_visibility(self):
        return self.__visibility
    
    #Aggiunto il get beta
    @property
    def get_beta(self):  
        return(self.__beta)
    
    #Aggiunto il get memory
    @property
    def get_memo(self):  
        return(self.__memory)


    #Aggiunto il get beta
    @property
    def get_beta(self):  
        return(self.__beta)


class WallFollowerAgent(Agent):
    def __init__(self, hand: str, visibility: int, **kargs) -> None:
        super().__init__(**kargs)
        self.__current_direction = None
        self.__last_position = None
        self.__facing_wall = False
        self.__hand = hand.lower()
        assert self.__hand in ["dx", "sx"], "hand deve essere 'dx' o 'sx'"

        self.__visibility = visibility
        assert self.__visibility > 0, "Visibility must be greater than 0"


    @property
    def class_type(self):
        return "WallFollower"

    def get_settings(self) -> dict:
        return {"hand": self.__hand}
    
    def get_visibility(self) -> int:
        return self.__visibility

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
        cardinal_order = [
            CardinalDirections.NORTH,
            CardinalDirections.EAST,
            CardinalDirections.SOUTH,
            CardinalDirections.WEST
        ]

        if self.__current_direction is None:
            self.__current_direction = CardinalDirections.EAST
            self.__last_position = current

        if self.__facing_wall:
            opposite_dir = {
                CardinalDirections.NORTH: CardinalDirections.SOUTH,
                CardinalDirections.EAST: CardinalDirections.WEST,
                CardinalDirections.SOUTH: CardinalDirections.NORTH,
                CardinalDirections.WEST: CardinalDirections.EAST
            }[self.__current_direction]
            
            for i, dir_info in enumerate(directions):
                if dir_info.orientation == opposite_dir:
                    self.__current_direction = opposite_dir
                    self.__facing_wall = False
                    return i

        current_idx = cardinal_order.index(self.__current_direction)
        
        if self.__hand == "dx":
            search_order = [
                cardinal_order[(current_idx + 1) % 4],  # dx
                cardinal_order[current_idx],            # Avanti
                cardinal_order[(current_idx - 1) % 4],  # sx
                cardinal_order[(current_idx + 2) % 4]   # Indietro
            ]
        else:
            search_order = [
                cardinal_order[(current_idx - 1) % 4],  # sx
                cardinal_order[current_idx],            # Avanti
                cardinal_order[(current_idx + 1) % 4],  # dx
                cardinal_order[(current_idx + 2) % 4]   # Indietro
            ]

        selected_dir = None
        for direction in search_order:
            for i, dir_info in enumerate(directions):
                if dir_info.orientation == direction:
                    selected_dir = direction
                    if direction != self.__current_direction:
                        self.__current_direction = direction
                    return i

        # Vicolo cieco
        if selected_dir is None:
            self.__facing_wall = True
            for i, dir_info in enumerate(directions):
                if dir_info.orientation == cardinal_order[(current_idx + 2) % 4]:
                    self.__current_direction = cardinal_order[(current_idx + 2) % 4]
                    return i

        return random_state.randint(0, len(directions))


class FrontFirstAgent(Agent):
    def __init__(self, visibility: int, turn_right_prob: float = 0.5, **kargs) -> None:
        super().__init__(**kargs)
        self.__current_direction = None
        self.__last_position = None
        self.__visibility = visibility
        assert self.__visibility > 0, "Visibility must be greater than 0"

        self.__turn_right_prob = turn_right_prob  # Probabilità di girare a destra quando incontra un ostacolo
        assert 0 <= self.__turn_right_prob <= 1, "Turn probability must be between 0 and 1"

    @property
    def class_type(self):
        return "FrontFirst"

    def get_settings(self) -> dict:
        return {'turn_right_prob': self.__turn_right_prob}
    
    def get_visibility(self) -> int:
        return self.__visibility

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
        cardinal_order = [
            CardinalDirections.NORTH,
            CardinalDirections.EAST,
            CardinalDirections.SOUTH,
            CardinalDirections.WEST
        ]

        if self.__current_direction is None:
            self.__current_direction = CardinalDirections.EAST
            self.__last_position = current

        # può andare dritto?
        for i, dir_info in enumerate(directions):
            if dir_info.orientation == self.__current_direction:
                return i  # Continua nella direzione corrente

        # non va dritto, allora decide se girare a destra o sinistra
        current_idx = cardinal_order.index(self.__current_direction)
        
        # Decide la direzione di rotazione in base alla probabilità, oraria/antioraria
        turn_right = random_state.random() < self.__turn_right_prob

        if turn_right:
            # ordine orario
            for offset in range(1, 5):  # 1-4 per coprire tutte le direzioni
                new_direction = cardinal_order[(current_idx + offset) % 4]
                for i, dir_info in enumerate(directions):
                    if dir_info.orientation == new_direction:
                        self.__current_direction = new_direction 
                        return i
        else:
            # ordine antiorario
            for offset in range(3, -1, -1):
                new_direction = cardinal_order[(current_idx + offset) % 4]
                for i, dir_info in enumerate(directions):
                    if dir_info.orientation == new_direction:
                        self.__current_direction = new_direction 
                        return i

        return random_state.randint(0, len(directions))


class TremauxAgent(Agent):
    def __init__(self, visibility: int, **kargs) -> None:
        super().__init__(**kargs)
        self.__marks = {}  # {(cell1, cell2): num_marks}
        self.__current_path = []  # path attuale
        self.__entrance = None 
        self.__visibility = visibility
        assert self.__visibility > 0, "Visibility must be greater than 0"

    @property
    def class_type(self):
        return "Tremaux"

    def get_settings(self) -> dict:
        return {}
    
    def get_visibility(self) -> int:
        return self.__visibility

    def __mark_passage(self, cell1: tuple, cell2: tuple):
        key = (min(cell1, cell2), max(cell1, cell2))
        self.__marks[key] = self.__marks.get(key, 0) + 1

    def __get_mark_count(self, cell1: tuple, cell2: tuple) -> int:
        key = (min(cell1, cell2), max(cell1, cell2))
        return self.__marks.get(key, 0)

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
        if not self.__entrance:
            self.__entrance = current
            self.__current_path.append(current)

        # Marca il passaggio dalla cella precedente a quella corrente
        if len(self.__current_path) > 1:
            prev_cell = self.__current_path[-2]
            self.__mark_passage(prev_cell, current)

        # Trova tutte le possibili mosse
        possible_moves = []
        for i, direction in enumerate(directions):
            next_cell = direction.adjacent_cell
            mark_count = self.__get_mark_count(current, next_cell)
            possible_moves.append((i, next_cell, mark_count))

        # Regola 3: Se incrocio senza segni (tranne eventualmente quello d'ingresso)
        unmarked = [m for m in possible_moves if m[2] == 0]
        if unmarked:
            chosen = random_state.choice([m[0] for m in unmarked])
            next_cell = directions[chosen].adjacent_cell
            self.__current_path.append(next_cell)
            return chosen

        # Regola 4a: Se il percorso d'ingresso ha 1 segno, torna indietro
        if len(self.__current_path) > 1:
            entry_mark = self.__get_mark_count(current, self.__current_path[-2])
            if entry_mark == 1:
                for i, direction in enumerate(directions):
                    if direction.adjacent_cell == self.__current_path[-2]:
                        self.__current_path.pop()
                        return i

        # Regola 4b: Scegli percorso con meno segni (0 o 1)
        min_mark = min(m[2] for m in possible_moves)
        candidates = [m for m in possible_moves if m[2] == min_mark]
        chosen = random_state.choice([m[0] for m in candidates])
        next_cell = directions[chosen].adjacent_cell
        self.__current_path.append(next_cell)
        return chosen

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["marks"] = {f"{k[0]}-{k[1]}": v for k, v in self.__marks.items()}
        return base_dict


class AdaptiveWalk(Agent):
    def __init__(self, visibility: int = 0.5, **kargs) -> None:
        super().__init__(**kargs)
        self.__visibility = visibility
        assert self.__visibility > 0, "Visibility must be greater than 0"
        self.__visited = set()
        self.__last_direction = None

    @property
    def class_type(self):
        return "AdaptiveWalk"

    def get_settings(self) -> dict:
        return {}

    def get_visibility(self) -> int:
        return self.__visibility

    def move(self, current: tuple, directions: list[DirectionInfo], random_state: np.random.RandomState) -> int:
        self.__visited.add(current) # memorizza quelle che visita

        # si cerca una cella mai visitata
        unvisited = [(i, d) for i, d in enumerate(directions) if d.adjacent_cell not in self.__visited]
        if unvisited:
            # ma tra quelle non visitate si sceglie quella che cambia meno la direzione!
            if self.__last_direction is not None:
                unvisited.sort(key=lambda x: x[1].orientation != self.__last_direction)
            chosen_idx, chosen_dir = unvisited[0]
            self.__last_direction = chosen_dir.orientation
            self.__visited.add(chosen_dir.adjacent_cell)
            return chosen_idx

        # se tutte le celle sono visitate (unvisited = [] array vuoto) si preferisce mantenere la direzione precedente
        if self.__last_direction is not None:
            for i, d in enumerate(directions):
                if d.orientation == self.__last_direction:
                    self.__visited.add(d.adjacent_cell)
                    return i

        # se si arriva qui allora si fa un "fallback" casuale (come RandomAgent)
        choice = random_state.randint(0, len(directions))
        self.__last_direction = directions[choice].orientation
        self.__visited.add(directions[choice].adjacent_cell)
        return choice

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "visited_cells": len(self.__visited)
        })
        return base


__all__ = ["Agent", "RandomAgent", "StaticAgent", "WallFollowerAgent", "FrontFirstAgent", "TremauxAgent", "AdaptiveWalk"]

#next
#modifiche per il beta e per il visibility, metodo per settare il beta in base alla formula 