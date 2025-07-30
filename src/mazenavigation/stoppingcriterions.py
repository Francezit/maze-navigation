
class StoppingCriteria():
    def __call__(self, time: float, path_len: int, cost: float) -> bool:
        raise Exception("Not implemented") 

class CostStoppingCriteria(StoppingCriteria):

    def __init__(self, max_cost: float) -> None:
        self.max_cost = max_cost

    def __call__(self, time: float, path_len: int, cost: float) -> bool:
        return cost >= self.max_cost
    

class TimeStoppingCriteria(StoppingCriteria):
    
    def __init__(self, max_time: float, max_cost: int = 10) -> None:
        self.max_time = max_time
        self.max_cost = max_cost

    def __call__(self, time: float, path_len: int, cost: float) -> bool:
        return time >= self.max_time
    
    
class ThresholdStoppingCriteria(StoppingCriteria):

    __inner_criterias: list
    __stop_if_at_least_one: bool

    def __init__(self, max_time: float = None, max_len_path: int = None, max_cost: float = None, stop_if_at_least_one: bool = True) -> None:
        self.max_time = max_time
        self.max_len_path = max_len_path
        self.max_cost = max_cost

        self.__stop_if_at_least_one = stop_if_at_least_one
        self.__inner_criterias = []

    def add_criteria(self, fun: "ThresholdStoppingCriteria"):
        self.__inner_criterias.append(fun)

    def __call__(self, time: float, path_len: int, cost: float) -> bool:
        condictions = []

        if self.max_time:
            condictions.append(time >= self.max_time)

        if self.max_len_path:
            condictions.append(path_len >= self.max_len_path)

        if self.max_cost:
            condictions.append(cost >= self.max_cost)

        for criteria in self.__inner_criterias:
            condictions.append(criteria(time, path_len, cost))

        if self.__stop_if_at_least_one:
            return any(condictions)
        else:
            return all(condictions)
            
    def get_stop_if_at_least_one(self):
        return self.__stop_if_at_least_one
    
    def get_time(self):
        return self.max_time

