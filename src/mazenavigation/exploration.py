import numpy as np
import os
import threading
import queue
import json
import time
import logging
import queue

from .interactionstrategies import *
from .utils import *
from.maze import Maze, MAZE_WALL_CODE
from.agent import Agent, DirectionInfo,StaticAgent
from.stoppingcriterions import *



class ExplorationState():

    @property
    def selected_direction(self):
        return self.available_directions[self.selected_direction_index]

    @property
    def delta_cost(self):
        return self.selected_path_cost-self.current_cost

    @property
    def num_explored_cells(self) -> int:
        return len(set(self.selected_path))

    @property
    def efficency(self):
        return self.num_explored_cells/len(self.selected_path)

    @property
    def selected_point(self):
        return self.selected_direction.adjacent_cell

    @property
    def selected_orientation(self):
        return self.selected_direction.orientation

    def __init__(self, point, available_directions: list[DirectionInfo], selected_direction_index: int,
                 selected_path: list, selected_path_cost: float, current_cost: float, time: float) -> None:
        self.point = point

        self.available_directions = [
            convert_dict_to_object(DirectionInfo, x) for x in available_directions
        ]
        self.selected_direction_index = selected_direction_index
        self.selected_path = selected_path
        self.selected_path_cost = selected_path_cost
        self.current_cost = current_cost
        self.time = time

    def __repr__(self) -> str:
        return f"{self.selected_orientation.name}: {self.point}->{self.selected_point} (+{self.delta_cost})"

    def to_dict(self):
        return {
            "point": self.point,
            "delta_cost": self.delta_cost,
            "selected_orientation": self.selected_orientation.name,
            "selected_point": self.selected_point,
            "available_directions": [x.to_dict() for x in self.available_directions],
            "selected_direction_index": self.selected_direction_index,
            "selected_path": self.selected_path,
            "selected_path_cost": self.selected_path_cost,
            "current_cost": self.current_cost,
            "time": self.time
        }


class ExplorationResult():

    @property
    def has_error(self):
        return self.err is not None

    @property
    def path_len(self):
        return len(self.path)

    @property
    def num_explored_cells(self) -> int:
        return len(set(self.path))

    @property
    def last_state(self) -> ExplorationState:
        return self.register[-1]

    @property
    def efficency(self):
        return self.num_explored_cells/self.path_len

   

    def __init__(self, agent_name: str, path: list[tuple], cost: float, cost_available: float, time: float,
                 is_killed: bool, register: list = None, err: str = None) -> None:
        self.agent_name = agent_name
        if path and len(path) > 0 and isinstance(path[0], list):
            self.path = [tuple(x) for x in path]
        else:
            self.path = path
        self.cost = cost
        self.cost_available = cost_available
        self.time = time
        self.is_killed = is_killed
        self.err = err
        
        
        if register:
            self.register = [
                convert_dict_to_object(ExplorationState, x)
                for x in register
            ]
        else:
            self.register = None

    def to_dict(self):
        result = {
            "agent_name": self.agent_name,
            "is_killed": self.is_killed,
            "path": self.path,
            "cost": self.cost,
            "cost_available": self.cost_available,
            "time": self.time,
            "err": self.err,
            "register": [x.to_dict() for x in self.register] if not self.register is None else None
        }
        return result
    

class InteractionExplorationResult(ExplorationResult):

    def __init__(self, agent_name: str, path: list[tuple], cost: float, cost_available: float, time: float,
                 is_killed: bool, register: list = None, err: str = None, final_beta: float = None,
                 beta_history: list = None, interaction_history: list = None) -> None:
        
        self.final_beta = final_beta
        self.beta_history = beta_history or []
        self.interaction_history = interaction_history or []

        super().__init__(agent_name, path, cost, cost_available, time, is_killed, register, err)


    def to_dict(self):
        obj=super().to_dict()
        obj.update({
            "beta_history": self.beta_history,
            "interaction_history": self.interaction_history,
            "final_beta": self.final_beta,  
        })
        return obj




def save_exploration_result(results: ExplorationResult, filename: str):
    from ._zipfilewrapper import json_dump_zip
    obj= results.to_dict()
    json_dump_zip(filename, obj)
    return obj


def load_exploration_result(item: str|dict) -> ExplorationResult:
    if isinstance(item, str):
        from ._zipfilewrapper import json_load_zip
        obj: dict = json_load_zip(item)
    elif isinstance(item, dict):
        obj = item
    else:
        raise Exception("Not valid exploration result")
    
    if "beta_history" in obj:
        results = convert_dict_to_object(InteractionExplorationResult, obj)
    else:
        results = convert_dict_to_object(ExplorationResult, obj)
    return results


class ExplorationResultCollection():
    _registers: list[ExplorationResult | str]
    __index: int
    __base_folder: str
    __use_cache: bool

    @property
    def count(self):
        return len(self._registers)

    @property
    def enable_cache(self):
        return self.__use_cache

    @enable_cache.setter
    def enable_cache(self, value: bool):
        self.__use_cache = value

    @property
    def results(self):
        return list(self)

    def __init__(self, registers: list[ExplorationResult | str], base_folder: str = None):
        self.__base_folder = base_folder
        self._registers = registers
        self.__index = 0
        self.__use_cache = False

    def __len__(self):
        return len(self._registers)

    def __iter__(self):
        return ExplorationResultCollection(self._registers, self.__base_folder)

    def __next__(self):
        if self.__index < len(self._registers):
            item = self.get_item(self.__index)
            self.__index += 1
            return item
        else:
            raise StopIteration

    def find_item_by_agent(self, agent_name: int) -> ExplorationResult:
        for i in range(len(self._registers)):
          if agent_name in self._registers[i]:
              return self.get_item(i)
        return None

    def get_item(self, index: int):
        item = self._registers[index]

        checked_item: ExplorationResult
        if isinstance(item, ExplorationResult):
            checked_item = item
        elif isinstance(item, str):
            if self.__base_folder:
                checked_item = load_exploration_result(
                    os.path.join(self.__base_folder, item)
                )
            else:
                checked_item = load_exploration_result(item)
        else:
            raise Exception("Not valid exploration result")

        if self.__use_cache:
            self._registers[index] = checked_item
        return checked_item

def save_exploration_results(data: ExplorationResultCollection | list[ExplorationResult], filename: str):
    if isinstance(data, list):
        data = ExplorationResultCollection(data)

    # Creare la directory se non esiste
    os.makedirs(os.path.dirname(filename), exist_ok=True) 

    with open(filename, "w") as fp:
        json.dump([
            x.to_dict() if isinstance(x, ExplorationResult) else x for x in data._registers
        ], fp)

def load_exploration_results(pathname: str, base_folder: str = None):
    registers: list[str]
    if os.path.isfile(pathname):
        with open(pathname, "r") as fp:
            registers =[
                load_exploration_result(x)
                for x in json.load(fp)
            ] 

        if base_folder is None:
            base_folder = os.path.dirname(pathname)
    else:
        registers = [
            f
            for f in os.listdir(pathname)
            if f.endswith(".zip")
        ]

        if base_folder is None:
            base_folder = pathname

    return ExplorationResultCollection(registers, base_folder)

    
class ExplorationCallback():
    def __call__(self, maze: Maze, agent: Agent, state: ExplorationState):
        pass

class Exploration():
    
    __maze: Maze
    __stop_criteria: StoppingCriteria
    __callback_function: ExplorationCallback
    __tick_unit: int
    __use_register: bool
    __random_state: np.random.RandomState

    @property
    def maze(self):
        return self.__maze

    def __init__(self, maze: Maze, stop_criteria: StoppingCriteria = None, callback_function: ExplorationCallback = None,
                 tick_unit=1, use_register: bool = False, random_state: np.random.RandomState = None, logger: logging.Logger = None) -> None:
        assert maze is not None and maze.is_loaded
        self.__maze = maze
        self.__stop_criteria = stop_criteria
        self.__callback_function = callback_function
        self.__tick_unit = tick_unit
        self.__use_register = use_register
       


        if random_state is None:
            self.__random_state = np.random.RandomState()
        else:
            self.__random_state = random_state

        if logger is None:
            self.__log_info = lambda msg: None
            self.__log_debug = lambda msg: None
        else:
            self.__log_info = logger.info
            self.__log_debug = logger.debug

    def parallel(self, agents: list[Agent], n_workers: int = None, results_folder: str = None, **kargs) -> ExplorationResultCollection:
        # check variables
        assert len(agents) >= 1, "At least one agent must be presented"
        assert all([isinstance(x, Agent) for x in agents]), "Not supported"
        
        # set global registers
        if results_folder:
            os.makedirs(results_folder)

        # set num workers
        if n_workers is None:
            n_workers = 100
        elif n_workers < 1:
            raise Exception("The number of workers must be greater than 1")
        n_workers = min(n_workers, len(agents))
        
        # define worker
        def worker_fun(tasks: queue.Queue):
            while not tasks.empty():
                item: dict = tasks.get()
                agent = item["agent"]
                register_filename = item.get("register_filename")
                
                try:
                    maze = self.__maze.matrix
                    enter_point = self.__maze.enter_point
                    exit_point = self.__maze.exit_point
                    stop_criteria = self.__stop_criteria
                    callback_function = self.__callback_function
                    tick_unit = self.__tick_unit
                    use_register = self.__use_register
                    random_state = self.__random_state

                    # print(f"{agent.name} enters in the maze from {enter_point}")

                    path = [enter_point]
                    cost_path = 0
                    current_point = enter_point
                    tick = 0
                    visibility_radius = agent.get_visibility()

                    # print(f"{agent.name} visibility radius: {visibility_radius}")

                    # Algoritmo Bresenham per linea di vista
                    def has_line_of_sight(start, end):
                        x0, y0 = start
                        x1, y1 = end
                        dx = abs(x1 - x0)
                        dy = abs(y1 - y0)
                        x, y = x0, y0
                        sx = -1 if x0 > x1 else 1
                        sy = -1 if y0 > y1 else 1
                        err = dx - dy

                        while x != x1 or y != y1:
                            if maze[x, y] >= MAZE_WALL_CODE:
                                return False
                            e2 = 2 * err
                            if e2 > -dy:
                                err -= dy
                                x += sx
                            if e2 < dx:
                                err += dx
                                y += sy
                        return True

                    register = []
                    while current_point != exit_point and (stop_criteria is None or not stop_criteria(tick, len(path), cost_path)):
                        points = get_indices_around_point(
                            matrix=maze,
                            point=current_point,
                            condiction=lambda w: not np.isnan(w) and w < MAZE_WALL_CODE,
                            remove_itself=True
                        )

                        directions = []
                        for point in points:
                            if point == enter_point:
                                continue

                            x, y = 0, 0
                            orientation = get_directions(current_point, point)
                            if orientation == CardinalDirections.NORTH:
                                x = -1
                            elif orientation == CardinalDirections.SOUTH:
                                x = 1
                            elif orientation == CardinalDirections.EAST:
                                y = 1
                            elif orientation == CardinalDirections.WEST:
                                y = -1
                            else:
                                raise Exception("Not supported")

                            p = point
                            lines = [p]
                            is_wall = False
                            while not is_wall:
                                p = (p[0]+x, p[1]+y)
                                if p[0] >= 0 and p[0] < maze.shape[0] and p[1] >= 0 and p[1] < maze.shape[1] and maze[p] < MAZE_WALL_CODE and p is not enter_point:
                                    lines.append(p)
                                else:
                                    is_wall = True

                            directions.append(
                                DirectionInfo(cells=lines,
                                            weights=[maze[t] for t in lines],
                                            orientation=orientation)
                            )

                        # Log
                        # print(f"\n{agent.name} - Current position: {current_point}")
                        # print(f"Visibility radius: {visibility_radius} cells")
                        # print("Available directions:")
                        # for direction in directions:
                        #     visible = [cell for cell in direction.cells if has_line_of_sight(current_point, cell)]
                        #     print(f"- {direction.orientation.name}: {len(visible)} visible cells")
                        #     if visible:
                        #         print(f"  Cells: {visible}")

                        # Distanza Manhattan e controllo uscita
                        exit_directions = [i for i, x in enumerate(directions) if exit_point in x.cells]
                        distance_to_exit = abs(current_point[0] - exit_point[0]) + abs(current_point[1] - exit_point[1])
                        exit_visible = (len(exit_directions) > 0 and 
                                    distance_to_exit <= visibility_radius and 
                                    has_line_of_sight(current_point, exit_point))

                        # print(f"\nExit visible: {exit_visible}")

                        if exit_visible:
                            # print(f"Exit distance: {distance_to_exit} cells")
                            # print(f"Exit direction: {directions[exit_directions[0]].orientation.name}")
                            selected_direction = directions[exit_directions[0]]
                            idx = exit_directions[0]
                            # print(f"{agent.name} sees exit at {exit_point} (distance: {distance_to_exit})")
                        else:
                            idx = agent.move(current_point, directions, random_state)
                            selected_direction = directions[idx]
                            # print(f"{agent.name} chose direction: {selected_direction.orientation.name}")

                        if tick_unit > 0:
                            tick += self.__tick_unit + (tick_unit * selected_direction.adjacent_weight)
                        else:
                            tick += self.__tick_unit

                        prev_cost = cost_path
                        cost_path += selected_direction.adjacent_weight
                        prev_point = current_point
                        current_point = selected_direction.adjacent_cell
                        path.append(current_point)

                        # print(f"Moved to {current_point} (cost: +{selected_direction.adjacent_weight})")

                        if use_register or callback_function:
                            state = ExplorationState(
                                point=prev_point,
                                available_directions=directions,
                                selected_direction_index=idx,
                                selected_path=path.copy(),
                                current_cost=prev_cost,
                                selected_path_cost=cost_path,
                                time=tick
                            )

                            if use_register:
                                register.append(state)

                            if callback_function:
                                callback_function(maze, agent, state)

                    is_killed = path[-1] != exit_point
                    
                    # if is_killed:
                    #     print(f"{agent.name} has been killed in {path[-1]}, cells visited {len(path)}, cost {cost_path}")
                    # else:
                    #     print(f"{agent.name} has exited in {tick} ticks, cells visited {len(path)}, cost {cost_path}")

                    result = ExplorationResult(
                        agent_name=agent.name,
                        path=path,
                        cost=cost_path,
                        cost_available=stop_criteria.max_cost-cost_path if stop_criteria else None,
                        time=tick,
                        is_killed=is_killed,
                        register=register
                    )

                    if register_filename:
                        save_exploration_result(result, register_filename)
                        item["result"] = os.path.basename(register_filename)
                    else:
                        item["result"] = result
                        
                except Exception as err:
                    item["result"] = ExplorationResult(
                        agent_name=agent.name,
                        path=None,
                        cost=None,
                        cost_available=None,
                        time=None,
                        is_killed=None,
                        register=None,
                        err=str(err)
                    )
                finally:
                    tasks.task_done()

        # create instances
        instances = []
        for agent in agents:
            item = kargs.copy()
            item.update({
                "agent": agent,
                "register_filename": os.path.join(results_folder, f"{agent.name}.zip") if results_folder else None
            })
            instances.append(item)

        worker_tasks = queue.Queue(len(instances))
        for exp in instances:
            worker_tasks.put(exp)

        # start processing
        if n_workers > 1:
            worker_list = []
            for _ in range(n_workers):
                p = threading.Thread(target=worker_fun, kwargs={"tasks": worker_tasks}, daemon=True)
                worker_list.append(p)
                p.start()
                
            for worker in worker_list:
                worker.join()
        else:
            worker_fun(worker_tasks)
        
        # get results
        return ExplorationResultCollection([item["result"] for item in instances])

    def single(self, agent: Agent) -> ExplorationResult:
        try:
            # set variables
            maze = self.__maze.matrix
            enter_point = self.__maze.enter_point
            exit_point = self.__maze.exit_point
            stop_criteria = self.__stop_criteria
            callback_function = self.__callback_function
            tick_unit = self.__tick_unit
            use_register = self.__use_register
            random_state = self.__random_state

            self.__log_info(f"{agent.name} enters in the maze from {enter_point}")

            # set temp variables
            st = time.time()
            path = [enter_point]
            cost_path = 0
            current_point = enter_point
            tick = 0

            visibility_radius = agent.get_visibility()

            # print(f"{agent.name} visibility radius: {visibility_radius}")

            # Funzione per verificare la linea di vista (algoritmo di Bresenham per i muri in mezzo alla vista)
            #TODO refactoring 
            def has_line_of_sight(start, end):
                """Verifica se c'è una linea di vista libera tra start e end"""
                x0, y0 = start
                x1, y1 = end
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                x, y = x0, y0
                sx = -1 if x0 > x1 else 1
                sy = -1 if y0 > y1 else 1
                err = dx - dy

                while x != x1 or y != y1:
                    if maze[x, y] >= MAZE_WALL_CODE:  # Se incontra un muro
                        return False
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy
                return True

            # start loop
            register = []
            while current_point != exit_point and (stop_criteria is None or not stop_criteria(tick, len(path), cost_path)):
                # compute adjacent points
                points: list = get_indices_around_point(
                    matrix=maze,
                    point=current_point,
                    condiction=lambda w: not np.isnan(w) and w < MAZE_WALL_CODE,
                    remove_itself=True
                )

                # compute the available directions
                directions: list[DirectionInfo] = []
                for point in points:
                    if point == enter_point:
                        continue  # this direction is discared as an agent can not be exit from the enter

                    x, y = 0, 0
                    orientation = get_directions(current_point, point)
                    if orientation == CardinalDirections.NORTH:
                        x = -1
                    elif orientation == CardinalDirections.SOUTH:
                        x = 1
                    elif orientation == CardinalDirections.EAST:
                        y = 1
                    elif orientation == CardinalDirections.WEST:
                        y = -1
                    else:
                        raise Exception("Not supported")

                    p = point
                    lines = [p]
                    is_wall = False
                    while not is_wall:
                        p = (p[0]+x, p[1]+y)
                        if p[0] >= 0 and p[0] < maze.shape[0] and p[1] >= 0 and p[1] < maze.shape[1] and maze[p] < MAZE_WALL_CODE and p is not enter_point:
                            lines.append(p)
                        else:
                            is_wall = True

                    directions.append(
                        DirectionInfo(cells=lines,
                                    weights=[maze[t] for t in lines],
                                    orientation=orientation)
                    )

                # Analisi delle celle visibili in ogni direzione
                visible_cells_per_direction = []
                for direction in directions:
                    visible_cells = []
                    for cell in direction.cells:
                        if has_line_of_sight(current_point, cell):
                            visible_cells.append(cell)
                    visible_cells_per_direction.append({
                        'orientation': direction.orientation.name,
                        'visible_cells': len(visible_cells),
                        'cells': visible_cells
                    })

                # log
                # print(f"\n{agent.name} - Posizione attuale: {current_point}")
                # print(f"Raggio di visibilità: {visibility_radius} celle")
                # print("Celle totali per direzione:")
                # for dir_info in visible_cells_per_direction:
                #     print(f"- {dir_info['orientation']}: {dir_info['visible_cells']} celle")
                #     if dir_info['visible_cells'] > 0:
                #         print(f"  Celle: {dir_info['cells']}")


                # Verifica se l'uscita è visibile (entro il raggio e in linea retta)
                exit_directions = [i for i, x in enumerate(directions) if exit_point in x.cells]

                # Distanza di manhattan
                distance_to_exit = abs(current_point[0] - exit_point[0]) + abs(current_point[1] - exit_point[1])

                exit_visible = (len(exit_directions) > 0 and distance_to_exit <= visibility_radius and has_line_of_sight(current_point, exit_point))


                # print(f"\nUscita visibile: {exit_visible}")

                if exit_visible:
                    # print(f"Distanza dall'uscita: {distance_to_exit} celle")
                    # print(f"Direzione dell'uscita: {directions[exit_directions[0]].orientation.name} \n\n")

                    # Se l'uscita è visibile, vai verso di essa
                    selected_direction = directions[exit_directions[0]]
                    idx = exit_directions[0]
                    self.__log_debug(f"{agent.name} sees exit at {exit_point} (distance: {distance_to_exit})")
                else:
                    # Altrimenti usa la strategia normale dell'agente
                    idx = agent.move(current_point, directions, random_state)
                    selected_direction = directions[idx]

                if tick_unit > 0:
                    tick += self.__tick_unit + (tick_unit * selected_direction.adjacent_weight)
                else:
                    tick += self.__tick_unit

                # update variables
                prev_cost = cost_path
                cost_path += selected_direction.adjacent_weight
                prev_point = current_point
                current_point = selected_direction.adjacent_cell
                path.append(current_point)

                if use_register or callback_function:
                    state = ExplorationState(
                        point=prev_point,
                        available_directions=directions,
                        selected_direction_index=idx,
                        selected_path=path.copy(),
                        current_cost=prev_cost,
                        selected_path_cost=cost_path,
                        time=tick
                    )

                    if use_register:
                        register.append(state)

                    if callback_function:
                        callback_function(maze, agent, state)

                    self.__log_debug(f"{agent.name} moves in {current_point} +{cost_path-prev_cost}, eff: {state.efficency}")

            is_killed = path[-1] != exit_point
            if is_killed:
                self.__log_info(f"{agent.name} has been killed in {path[-1]}, cells visited {len(path)}, cost {cost_path} ")
            else:
                self.__log_info(f"{agent.name} has exited in {tick} tick, cells visited {len(path)}, cost {cost_path}")
        
            return ExplorationResult(
                agent_name=agent.name,
                path=path,
                cost=cost_path,
                cost_available=stop_criteria.max_cost-cost_path if stop_criteria else None,
                time=tick,
                is_killed=is_killed,
                register=register
            )
        except Exception as err:
            return ExplorationResult(
                agent_name=agent.name,
                path=None,
                cost=None,
                cost_available=None,
                time=None,
                is_killed=None,
                register=None,
                err=str(err)
            )

    def interaction(
        self, 
        agents: list[StaticAgent], 
        results_folder: str = None,
        update_beta_func:InteractionStrategy=None,
        beta_update_interval: int = 100
    ) -> ExplorationResultCollection:
        """
        Esegue l'interazione tra gli agenti nel labirinto.
        
        Args:
            agents: Lista di agenti che partecipano all'esplorazione
            results_folder: Cartella dove salvare i risultati (opzionale)
            update_beta_func: Funzione per aggiornare il beta degli agenti
            beta_update_interval: Frequenza di aggiornamento del beta (in step)
        
        Returns:
            ExplorationResultCollection con i risultati dell'esplorazione
        """
        update_beta_func = update_beta_func or DynamicInteractionStrategy
        assert len(agents) >= 1, "Deve esserci almeno un agente"
        assert all(isinstance(x, Agent) for x in agents), "Lista agenti non valida"

        stop_criteria = self.__stop_criteria  

        # Inizializzazione strutture dati per tracciamento agenti
        agent_positions = {agent: self.__maze.enter_point for agent in agents}
        agent_paths = {agent: [self.__maze.enter_point] for agent in agents}
        agent_costs = {agent: 0 for agent in agents}
        agent_states = {agent: [] for agent in agents}
        agent_time_list = {agent: 0 for agent in agents}
        agent_errors = {agent: None for agent in agents}  # Dizionario per tracciare errori
        active_agents = set(agents)
        
        # Nuove strutture per tracciare beta e interazioni
        agent_beta_history = {agent: [agent.get_beta] for agent in agents}
        agent_interaction_history = {agent: [] for agent in agents}

        step = 0

        while active_agents:
            step += 1

            # Aggiornamento beta ogni N step
            if step % beta_update_interval == 0:
                for agent in active_agents:
                    try:
                        # Trova agenti visibili con costo minore
                        visible_agents = [
                            other_agent for other_agent in active_agents 
                            if other_agent != agent 
                            and self.__is_agent_visible(agent, other_agent, agent_positions)
                            and agent_costs[other_agent] < agent_costs[agent]
                        ]
                        
                        if visible_agents:
                            old_beta = agent.get_beta
                            new_beta = update_beta_func(agent, visible_agents, agent_costs)
                            agent.set_beta(new_beta)
                            
                            # Registra la storia del beta
                            agent_beta_history[agent].append(new_beta)
                            
                            # Registra l'interazione
                            interaction_info = {
                                "step": step,
                                "time": agent_time_list[agent],
                                "old_beta": old_beta,
                                "new_beta": new_beta,
                                "visible_agents": [
                                    {
                                        "name": a.name,
                                        "position": agent_positions[a],
                                        "cost": agent_costs[a],
                                        "beta": a.get_beta
                                    } 
                                    for a in visible_agents
                                ]
                            }
                            agent_interaction_history[agent].append(interaction_info)
                            
                            print(f"Step {step}: {agent.name} beta updated {old_beta}→{new_beta} based on {len(visible_agents)} agents")
                    except Exception as e:
                        agent_errors[agent] = str(e)
                        print(f"Error updating beta for agent {agent.name}: {str(e)}")
                        continue

            # Log dello stato corrente
            print(f"\nStep {step}:")
            
            for agent in active_agents:
                print(f"  {agent.name}: Pos={agent_positions[agent]}, Time={agent_time_list[agent]:.2f}, "
                    f"Cost={agent_costs[agent]:.2f}, Beta={agent.get_beta:.2f}, "
                    f"Steps={len(agent_paths[agent])}")

            # Movimento per ogni agente attivo
            for agent in active_agents.copy():
                try:
                    current_pos = agent_positions[agent]
                    
                    # Controllo criterio di stop individuale
                    if stop_criteria(agent_time_list[agent], len(agent_paths[agent]), agent_costs[agent]):
                        print(f"Agent {agent.name} stopped: Time={agent_time_list[agent]:.2f}, "
                            f"Cost={agent_costs[agent]:.2f}, Steps={len(agent_paths[agent])}")
                        active_agents.remove(agent)
                        continue

                    # Calcolo possibili direzioni
                    maze = self.__maze.matrix
                    points = get_indices_around_point(
                        matrix=maze,
                        point=current_pos,
                        condiction=lambda w: not np.isnan(w) and w < MAZE_WALL_CODE,
                        remove_itself=True
                    )

                    # Costruzione informazioni direzioni
                    directions = []
                    for point in points:
                        if point == self.__maze.enter_point:
                            continue

                        x, y = 0, 0
                        orientation = get_directions(current_pos, point)
                        if orientation == CardinalDirections.NORTH:
                            x = -1
                        elif orientation == CardinalDirections.SOUTH:
                            x = 1
                        elif orientation == CardinalDirections.EAST:
                            y = 1
                        elif orientation == CardinalDirections.WEST:
                            y = -1
                        else:
                            raise Exception("Direzione non supportata")

                        p = point
                        lines = [p]
                        is_wall = False
                        while not is_wall:
                            p = (p[0]+x, p[1]+y)
                            if 0 <= p[0] < maze.shape[0] and 0 <= p[1] < maze.shape[1] and maze[p] < MAZE_WALL_CODE and p is not self.__maze.enter_point:
                                lines.append(p)
                            else:
                                is_wall = True

                        directions.append(
                            DirectionInfo(cells=lines, weights=[maze[t] for t in lines], orientation=orientation)
                        )

                    if not directions:
                        print(f"Agent {agent.name} has no valid directions!")
                        active_agents.remove(agent)
                        continue

                    # Selezione direzione (priorità all'uscita se visibile)
                    exit_directions = [d for d in directions if self.__maze.exit_point in d.get_visibile_cells(agent.get_visibility)]
                    if exit_directions:
                        selected_direction = exit_directions[0]
                        move_index = 0
                    else:
                        move_index = agent.move(current_pos, directions, self.__random_state)
                        selected_direction = directions[move_index]

                    # Aggiornamento stato agente
                    agent_time_list[agent] += self.__tick_unit
                    new_pos = selected_direction.adjacent_cell
                    agent_positions[agent] = new_pos
                    agent_paths[agent].append(new_pos)
                    agent_costs[agent] += selected_direction.adjacent_weight

                    state = ExplorationState(
                        point=current_pos,
                        available_directions=directions,
                        selected_direction_index=move_index,
                        selected_path=agent_paths[agent].copy(),
                        selected_path_cost=agent_costs[agent],
                        current_cost=agent_costs[agent] - selected_direction.adjacent_weight,
                        time=agent_time_list[agent]
                    )
                    agent_states[agent].append(state)

                    # Controllo uscita
                    if new_pos == self.__maze.exit_point:
                        active_agents.remove(agent)
                        print(f"Agent {agent.name} reached exit! Time={agent_time_list[agent]:.2f}, "
                            f"Cost={agent_costs[agent]:.2f}, Steps={len(agent_paths[agent])}")

                except Exception as e:
                    agent_errors[agent] = str(e)
                    print(f"Agent {agent.name} failed with error: {str(e)}")
                    active_agents.remove(agent)
                    continue

        # Raccolta risultati finali
        agents_exited = [agent for agent in agents if agent_positions[agent] == self.__maze.exit_point]
        agents_dead = [agent for agent in agents if agent not in agents_exited]

        print("\n=== FINAL RESULTS ===")
        print(f"Total steps: {step}")
        print(f"Agents exited: {len(agents_exited)}")
        print(f"Agents stopped: {len(agents_dead)}")
        print("Details:")
        for agent in agents:
            status = "EXITED" if agent in agents_exited else "STOPPED"
            error_info = f", Error={agent_errors[agent]}" if agent_errors[agent] else ""
            print(f"{agent.name}: {status}, Time={agent_time_list[agent]:.2f}, "
                f"Cost={agent_costs[agent]:.2f}, Beta={agent.get_beta:.2f}, "
                f"Steps={len(agent_paths[agent])}{error_info}")

        # Creazione risultati con le nuove informazioni
        results = [
            InteractionExplorationResult(
                agent_name=agent.name,
                path=agent_paths[agent],
                cost=agent_costs[agent],
                cost_available=getattr(stop_criteria, 'max_cost', float('inf')) - agent_costs[agent],
                time=agent_time_list[agent],
                is_killed=agent_positions[agent] != self.__maze.exit_point,
                register=None,
                final_beta=agent.get_beta,
                beta_history=agent_beta_history[agent],
                interaction_history=agent_interaction_history[agent],
                err=agent_errors[agent]  # Aggiunto il campo errori
            )
            for agent in agents
        ]

        # Salvataggio su file se richiesto
        if results_folder:
            os.makedirs(results_folder, exist_ok=True)
            for result in results:
                filename = os.path.join(results_folder, f"{result.agent_name}.zip")
                save_exploration_result(result, filename)

        return ExplorationResultCollection(results)
  
    def __is_agent_visible(self, agent: Agent, other_agent: Agent, agent_positions: dict) -> bool:
        """
        Controlla se un altro agente è visibile dall'agente corrente, tenendo conto delle pareti.
        """
        current_pos = agent_positions[agent]
        other_pos = agent_positions[other_agent]

        # Calcola la distanza di Manhattan
        distance = abs(current_pos[0] - other_pos[0]) + abs(current_pos[1] - other_pos[1])

        #Se la distanza è maggiore di x celle, l'agente non è visibile
        if distance > 5:
            return False

        # Se la distanza è minore o uguale a x ma maggiore della visibilità dell'agente, escludilo comunque
        if distance > agent.get_visibility:
            return False

        # Usa l'algoritmo di Bresenham (raster) per tracciare una linea tra le due posizioni
        x0, y0 = current_pos
        x1, y1 = other_pos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while (x0, y0) != (x1, y1):
            # Se la cella corrente è un muro, l'agente non è visibile
            if self.__maze.matrix[x0, y0] >= MAZE_WALL_CODE:  
                return False

            # Passa alla prossima cella lungo la linea
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        # Se non ci sono muri lungo il percorso, l'agente è visibile
        return True

    ###############################################CHECK###############################################

  
__all__ = [
    "DirectionInfo", "ExplorationState", "ExplorationResult", "ExplorationResultCollection",
    "save_exploration_result", "load_exploration_result",
    "load_exploration_results", "save_exploration_results",
    "ExplorationCallback", "Exploration"
]
