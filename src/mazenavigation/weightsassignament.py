import numpy as np


class WeightsAssignament:
    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        raise Exception("Not implemented")


class StaticWeightsAssignament(WeightsAssignament):

    def __init__(self, weight_value: float = 0) -> None:
        super().__init__()
        assert weight_value >= 0 and weight_value < 1
        self.__weight_value = weight_value

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = np.ones(maze_shape)*self.__weight_value
        return w


class UniformWeightsAssignament(WeightsAssignament):

    def __init__(self, weight_factor: float = 0):
        super().__init__()
        assert weight_factor < 1
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = rdn.random_sample(maze_shape)
        w *= self.__weight_factor
        return w


class GaussianWeightsAssignament(WeightsAssignament):

    def __init__(self, mean: float = 0.5, std: float = 0.1, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        self.__mean = mean
        self.__std = std
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = rdn.normal(self.__mean, self.__std, maze_shape)
        w = (w-np.min(w))/(np.max(w)-np.min(w))
        w *= self.__weight_factor
        return w


class BinomialWeightsAssignament(WeightsAssignament):

    def __init__(self, num: int = 10, prob: float = 0.6, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        assert isinstance(num, int) and num >= 0
        assert isinstance(prob, float) and prob >= 0 and prob <= 1
        self.__num = num
        self.__prob = prob
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = rdn.binomial(self.__num, self.__prob, maze_shape)
        w = (w-np.min(w))/(np.max(w)-np.min(w))
        w *= self.__weight_factor
        return w


class CenterWeightsAssignament(WeightsAssignament):

    def __init__(self, radius: float = 0.7, noise_level=0.1, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        assert radius > 0 and radius <= 1
        self.__radius = radius
        self.__noise_level = noise_level
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = generate_matrix_with_expansion(
            n=maze_shape[0],
            noise_level=self.__noise_level,
            radius=self.__radius,
            rdn=rdn
        )
        w *= self.__weight_factor
        return w


class EdgeWeightsAssignament(WeightsAssignament):

    def __init__(self, radius: float = 0.7, noise_level: float = 0.1, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        assert radius > 0 and radius <= 1
        self.__radius = radius
        self.__noise_level = noise_level
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = generate_matrix_with_expansion(
            n=maze_shape[0],
            noise_level=self.__noise_level,
            radius=self.__radius,
            rdn=rdn
        )
        w = (1-w) * self.__weight_factor
        return w


class MixWeightsAssignament(WeightsAssignament):

    def __init__(self, weight_strategies: list[WeightsAssignament], grades: list[float] = None, weight_factor: float = 0.7) -> None:
        super().__init__()
        assert weight_factor < 1
        assert len(weight_strategies) > 1
        if grades is None:
            grades = [1/len(weight_strategies) for _ in len(weight_strategies)]
        else:
            assert len(grades) == len(weight_strategies)
            assert sum(grades) == 1
        self.__weight_strategies = weight_strategies
        self.__grades = grades
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):

        w = None
        for strategy, grade in zip(self.__weight_strategies, self.__grades):
            if w is None:
                w = grade * strategy(maze_shape, rdn, **kargs)
            else:
                w += grade*strategy(maze_shape, rdn, **kargs)

        w *= self.__weight_factor
        return w


class DiagonalWeightsAssignament(WeightsAssignament):

    def __init__(self, radius: float = 0.7, noise_level: float = 0.1, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        assert radius > 0 and radius <= 1
        self.__radius = radius
        self.__noise_level = noise_level
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        w = generate_matrix_with_diagonal(
            n=maze_shape[0],
            noise_level=self.__noise_level,
            radius=self.__radius,
            rdn=rdn
        )
        w *= self.__weight_factor
        return w


class PathWeightsAssignament(WeightsAssignament):
    def __init__(self, radius: float = 0.7, noise_level: float = 0.1, weight_factor: float = 0.7):
        super().__init__()
        assert weight_factor < 1
        assert radius > 0 and radius <= 1
        self.__radius = radius
        self.__noise_level = noise_level
        self.__weight_factor = weight_factor

    def __call__(self, maze_shape: tuple, rdn: np.random.RandomState, **kargs):
        exit_point = kargs.get("exit")

        w = generate_matrix_with_target_point(
            n=maze_shape[0],
            point=exit_point,
            noise_level=self.__noise_level,
            radius=self.__radius,
            rdn=rdn
        )
        w *= self.__weight_factor
        return w


def generate_matrix_with_diagonal(n: int, noise_level=0.1, radius=0.5, rdn: np.random.RandomState = None):
    '''
    generates an nxn matrix with values between 0 and 1, 
    where the smaller numbers are in the diagonal of the matrix.
    '''

    if rdn is None:
        rdn = np.random

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance = abs(i - j)/(n-1)
            matrix[i, j] = np.sqrt(distance*radius)

    noise = rdn.uniform(-noise_level, noise_level, (n, n))
    matrix_with_noise = np.clip(matrix + noise, 0, 1)

    return matrix_with_noise


def generate_matrix_with_expansion(n: int, noise_level=0.1, radius=0.5, rdn: np.random.RandomState = None):
    '''
    generates an nxn matrix with values between 0 and 1, 
    where the greater numbers are in the center of the matrix and the lower numbers are on the border. 
    '''

    if rdn is None:
        rdn = np.random

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_to_center = np.sqrt((i - n/2)**2 + (j - n/2)**2) / (n/2)
            matrix[i, j] = 1 - distance_to_center * radius

    noise = rdn.uniform(-noise_level, noise_level, (n, n))
    matrix_with_noise = np.clip(matrix + noise, 0, 1)

    return matrix_with_noise


def calculate_distance_matrix(n, point):
    d = n*np.sqrt(2)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.sqrt(
                (i - point[0])**2 + (j - point[1])**2)/d
    return distance_matrix


def generate_matrix_with_target_point(n, point: tuple, noise_level=0.1, radius=0.5, rdn: np.random.RandomState = None):
    if rdn is None:
        rdn = np.random

    distance_matrix = calculate_distance_matrix(n, point)

    noise = rdn.uniform(-noise_level, noise_level, (n, n))
    matrix_with_noise = np.clip(
        distance_matrix*radius*1.5 + noise, 0, 1)

    return matrix_with_noise


__weightsassignaments = {
    "StaticWeightsAssignament": StaticWeightsAssignament,
    "UniformWeightsAssignament": UniformWeightsAssignament,
    "DiagonalWeightsAssignament": DiagonalWeightsAssignament,
    "GaussianWeightsAssignament": GaussianWeightsAssignament,
    "BinomialWeightsAssignament": BinomialWeightsAssignament,
    "CenterWeightsAssignament": CenterWeightsAssignament,
    "EdgeWeightsAssignament": EdgeWeightsAssignament,
    "PathWeightsAssignament": PathWeightsAssignament
}


def get_weightsassignament_strategies():
    return list(__weightsassignaments.keys())


def get_weightsassignament(strategy: str, **kargs) -> WeightsAssignament:
    strategy = __weightsassignaments.get(strategy, None)
    if strategy is None:
        raise Exception(f"Strategy not found {strategy}")
    return strategy(**kargs)


def combine_weightsassignaments(strategies: list[WeightsAssignament], p: list[float] = None, weight_factor: float = 1) -> WeightsAssignament:
    return MixWeightsAssignament(strategies, p, weight_factor)


__all__ = [
    "WeightsAssignament", "get_weightsassignament",
    "get_weightsassignament_strategies", "combine_weightsassignaments"
] + list(__weightsassignaments.keys())
