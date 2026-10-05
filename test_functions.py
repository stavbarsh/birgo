import torch

class Ackley:

    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-32.768, 32.768)] * self.dim
        # self.bounds = [(-1, 1)] * self.dim

    def fun(self, x):
        return (-20 * torch.exp(-0.2 * torch.sqrt(1 / self.dim * torch.sum(x ** 2)))
                - torch.exp(1 / self.dim * torch.sum(torch.cos(2.0 * torch.pi * x))) + torch.e + 20)


class AckleyN2:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = -200
        self.bounds = [(-32, 32)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return -200 * torch.exp(-0.2 * torch.sqrt(x ** 2 + y ** 2))


class Adjiman:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([2.0, 0.10578])]
        self.fglob = -2.02180678
        self.bounds = [(-1, 2), (-1, 1)]

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return (torch.cos(x) * torch.sin(y)) - (x / (y ** 2.0 + 1.0))


class AMGM:

    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.ones(self.dim)] # Infinit number of global minimum
        self.fglob = 0
        self.bounds = [(0, 10)] * self.dim

    def fun(self, x):
        a = torch.sum(x) / self.dim
        b = torch.prod(x) ** (1 / self.dim)
        return (a - b) ** 2


class BartelsConn:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 1
        self.bounds = [(-500, 500)] * self.dim

    def fun(self, xy):
        x1, x2 = xy[0], xy[1]
        return torch.abs(x1 ** 2 + x2 ** 2 + x1 * x2) + torch.abs(torch.sin(x1)) + torch.abs(torch.cos(x2))


class Beale:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([3, 0.5])]
        self.fglob = 0
        self.bounds = [(-4.5, 4.5)] * self.dim
        self.log = True

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return ((1.500 - x + x * y) ** 2 +
                (2.250 - x + x * y ** 2) ** 2 +
                (2.625 - x + x * y ** 3) ** 2)


class BentCigar:

    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim
        self.log = True

    def fun(self, x):
        return x[0] ** 2 + 1e6 * torch.sum(x[1:] ** 2)


class Bird:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([4.70105, 3.15294])]
        self.fglob = -106.76453
        self.bounds = [(-2 * torch.pi, 2 * torch.pi), (-2 * torch.pi, 2 * torch.pi)]

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return torch.sin(x) * torch.exp((1 - torch.cos(y)) ** 2) + torch.cos(y) * torch.exp((1 - torch.sin(x)) ** 2) + (
                x - y) ** 2


class BohachevskyN1:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return x ** 2 + 2 * y ** 2 - 0.3 * torch.cos(3.0 * torch.pi * x) - 0.4 * torch.cos(4 * torch.pi * y) + 0.7


class BohachevskyN2:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return x ** 2 + 2 * y ** 2 - 0.3 * torch.cos(3 * torch.pi * x) * torch.cos(4 * torch.pi * y) + 0.3


class BohachevskyN3:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return x ** 2 + 2 * y ** 2 - 0.3 * torch.cos(3 * torch.pi * x + 4 * torch.pi * y) + 0.3


class Booth:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([1, 3])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim
        self.log = True

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


class Branin:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([torch.pi, 2.275])] # one of the global minimas
        self.fglob = 0.397887
        self.bounds = [(-5, 10), (0, 15)]

    def fun(self, xy):
        a = 1
        b = 5.1 / (4 * torch.pi ** 2)
        c = 5 / torch.pi
        r = 6
        s = 10
        t = 1 / (8 * torch.pi)
        x, y = xy[0], xy[1]
        return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * torch.cos(x) + s


class Brent:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([-10, -10])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x, y, = xy[0], xy[1]
        return (x + 10.0) ** 2.0 + (y + 10.0) ** 2.0 + torch.exp(-x ** 2.0 - y ** 2.0)


class Brown:

    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-1, 4)] * self.dim

    def fun(self, x):
        return sum(
            [((x[i] ** 2) ** (x[i + 1] ** 2 + 1.0)) + ((x[i + 1] ** 2) ** (x[i] + 1.0)) for i in range(self.dim - 1)])


class BukinN6:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([-10, 1])]
        self.fglob = 0
        self.bounds = [(-15, -5), (-3, 3)]

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 100 * torch.sqrt(torch.abs(y - 0.01 * x ** 2)) + 0.01 * torch.abs(x + 10)


class CamelHump3:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-5, 5)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 2.0 * x ** 2 - 1.05 * x ** 4 + (x ** 6 / 6.0) + x * y + y ** 2


class CamelHump6:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0.0898, -0.7126])] # One of the global minima
        self.fglob = -1.0316
        self.bounds = [(-5, 5)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        a = (4 - (2.1 * x ** 2) + (x ** 4) / 3.0) * x ** 2
        b = x * y
        c = (-4 + (4 * y ** 2)) * y ** 2
        return a + b + c


class CarromTable:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([9.646157266348881, 9.646134286497169])]  # One of the local minimas
        self.fglob = -24.15681551650653
        self.bounds = [(-10, 10)] * self.dim  # Assuming bounds based on typical ranges

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return ((torch.exp(torch.abs(1 - (torch.sqrt(x ** 2 + y ** 2) / torch.pi))) * torch.cos(x) * torch.cos(y)) ** 2
                / -30.0)


class Chichinadze:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([6.189866586965680, 0.5])]
        self.fglob = -42.9443870
        self.bounds = [(-30, 30)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return x ** 2 - 12 * x + 11 + 10 * torch.cos(torch.pi * x / 2) + 8 * torch.sin(
            5 * torch.pi * x / 2) - 1.0 / 5 ** 0.5 * torch.exp(-((y - 0.5) ** 2) / 2)


class ChungReynolds:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim
        self.log = True

    def fun(self, x):
        return torch.sum(x ** 2) ** 2


class Colville:
    def __init__(self, dim=4):
        self.dim = 4
        self.N = self.dim
        self.global_optimum = [torch.ones(self.dim)]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x1, x2, x3, x4 = xy[0], xy[1], xy[2], xy[3]
        a = 100 * (x1 ** 2 - x2) ** 2
        b = (x1 - 1) ** 2
        c = (x3 - 1) ** 2
        d = 90 * (x3 ** 2 - x4) ** 2
        e = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        f = 19.8 * (x2 - 1) * (x4 - 1)
        return a + b + c + d + e + f


class CosineMixture:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [-torch.ones(self.dim)]
        self.fglob = -0.9 * self.dim
        self.bounds = [(-1, 1)] * self.dim

    def fun(self, x):
        return 0.1 * torch.sum(torch.cos(5 * torch.pi * x)) - torch.sum(x ** 2)


class Csendes:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-1, 1)] * self.dim

    def safe_div(self, n, d):
        """Safe Division - Specifically for handling the case when the denominator is 0.
        """
        if n / d == torch.inf:
            return torch.tensor(0)
        else:
            return n / d

    def fun(self, x):
        return sum([v ** 6 * (2 + torch.sin(self.safe_div(1, v))) for v in x])


class Cube:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.ones(self.dim)]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 100 * (y - x ** 3) ** 2 + (1 - x) ** 2


class Damavandi:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([2, 2])]
        self.fglob = 0
        self.bounds = [(0, 14)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        if x == 2 and y == 2: return torch.tensor(0.0)
        n = torch.sin(torch.pi * (x - 2)) * torch.sin(torch.pi * (y - 2))
        d = torch.pi ** 2 * (x - 2) * (y - 2)
        a = 1.0 - (torch.abs(n / d)) ** 5
        b = 2.0 + (x - 7) ** 2 + 2 * (y - 7) ** 2
        return a * b


class DeckkersAarts:

    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, 15])]
        self.fglob = -24771.093749
        self.bounds = [(-20, 20)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 10 * 5 * x ** 2 + y ** 2 - (x ** 2 + y ** 2) ** 2 + 10 ** (-5) * (x ** 2 + y ** 2) ** 4


class DixonPrice:

    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.tensor([2 ** (-((2 ** (i + 1)) - 2) / (2 ** (i + 1))) for i in range(self.dim)])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, x):
        return (x[0] - 1.0) ** 2.0 + sum([i * (2.0 * x[i] ** 2.0 - x[i - 1]) ** 2.0 for i in range(1, self.dim)])


class DropWave:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = -1
        self.bounds = [(-5.12, 5.12)] * self.dim

    def fun(self, x):
        return -(1 + torch.cos(12 * torch.sqrt(torch.sum(x ** 2)))) / (0.5 * torch.sum(x ** 2) + 2)


class Easom:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([torch.pi, torch.pi])]
        self.fglob = -1
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return -torch.cos(x) * torch.cos(y) * torch.exp(-(x - torch.pi) ** 2 - (y - torch.pi) ** 2)


class Eggholder:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([512, 404.2319])]
        self.fglob = -959.6407
        self.bounds = [(-512, 512)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return (-(y + 47) * torch.sin(torch.sqrt(torch.abs((x / 2.0) + (y + 47)))) -
                x * torch.sin(torch.sqrt(torch.abs(x - (y + 47)))))


class Exponential:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = -1
        self.bounds = [(-1, 1)] * self.dim

    def fun(self, x):
        return -torch.exp(-0.5 * torch.sum(x ** 2))


class FreudensteinRoth:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([5, 4])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        a = (x - 13 + ((5 - y) * y - 2) * y) ** 2
        b = (x - 29 + ((y + 1) * y - 14) * y) ** 2
        return a + b


class GoldsteinPrice:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, -1])]
        self.fglob = 3
        self.bounds = [(-2, 2)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return ((1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) *
                (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)))


class Griewank:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, x):
        i = torch.arange(1, self.dim + 1)
        return torch.sum(x ** 2 / 4000) - torch.prod(torch.cos(x / torch.sqrt(i))) + 1


class Himmelblau:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([3.0, 2.0])]  # One of the global minimum points
        self.fglob = 0
        self.bounds = [(-5, 5)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


class HolderTable:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([8.05502, 9.66459])]  # One of the global minimum points
        self.fglob = -19.2085
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return -torch.abs(
            torch.sin(x) * torch.cos(y) * torch.exp(torch.abs(1 - (torch.sqrt(x ** 2 + y ** 2) / torch.pi))))


class Hosaki:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([4, 2])]
        self.fglob = -2.3458
        self.bounds = [(0, 5), (0, 6)]

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return (1 - 8 * x + 7 * x ** 2 - 7.0 / 3.0 * x ** 3 + 0.25 * x ** 4) * y ** 2 * torch.exp(-y)


class Keane:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, 1.3932490])] # One of the global minimum points
        self.fglob = 0.6736675
        self.bounds = [(0, 10)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        n = (torch.sin(x - y) ** 2) * (torch.sin(x + y) ** 2)
        d = torch.sqrt(x ** 2 + y ** 2)
        return n / d


class Leon:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([1, 1])]
        self.fglob = 0
        self.bounds = [(-1.2, 1.2)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


class LeviN13:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([1, 1])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim
        self.log = True

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return torch.sin(3.0 * torch.pi * x) ** 2 + (x - 1) ** 2 * (1 + torch.sin(3.0 * torch.pi * y) ** 2) + (
                y - 1) ** 2 * (1 + torch.sin(2.0 * torch.pi * y) ** 2)


class Matyas:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, 0])]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


class McCormick:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([-0.5471975602214493, -1.547197559268372])]
        self.fglob = -1.913222954981037
        self.bounds = [(-1.5, 4), (-3, 3)]

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


class Michalewicz:
    def __init__(self, dim):
        self.dim = 2
        self.m = 10
        self.N = self.dim
        self.global_optimum = [torch.tensor([2.20290555, 1.570796])]  # One of the global minimum points for 2D
        self.fglob = -1.8013
        self.bounds = [(0, torch.pi)] * self.dim

    def fun(self, x):
        i = torch.arange(1, self.dim + 1)
        return -torch.sum(torch.sin(x) * (torch.sin(i * x ** 2 / torch.pi)) ** (2 * self.m))


class Parsopoulos:
    def __init__(self, dim):
        self.dim = 2
        # Note: Infinite global minimum points exist.
        # The exact coordinates depend on k and l.
        self.N = self.dim
        self.global_optimum = [torch.tensor([torch.pi / 2.0, torch.pi])]
        self.fglob = 0
        self.bounds = [(-5, 5)] * self.dim

    def fun(self, x):
        return torch.cos(x[0]) ** 2 + torch.sin(x[1]) ** 2


class PenHolder:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([9.6461677, 9.6461676])]  # One of the global minimum points
        self.fglob = -0.9635348
        self.bounds = [(-11, 11)] * self.dim

    def fun(self, xy):
        return -torch.exp(-(torch.abs(torch.cos(xy[0]) * torch.cos(xy[1]) * torch.exp(
            torch.abs(1 - torch.sqrt(xy[0] ** 2 + xy[1] ** 2) / torch.pi)))) ** -1)


class Qing:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([i ** 0.5 for i in range(1, self.dim + 1)])]
        self.fglob = 0
        self.bounds = [(-500, 500)] * self.dim

    def fun(self, x):
        return sum([(v ** 2 - (i + 1)) ** 2 for i, v in enumerate(x)])


class Rastrigin:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-5.12, 5.12)] * self.dim

    def fun(self, x):
        return 10.0 * self.dim + torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * torch.pi * x))


class Rosenbrock:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.ones(self.dim)]
        self.fglob = 0
        # self.bounds = [(-30, 30)] * self.dim
        self.bounds = [(-2.048, 2.048)] * self.dim
        self.log = True

    def fun(self, x):
        total = 0
        for i in range(self.dim - 1):
            total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return total


class Salomon:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, x):
        return 1 - torch.cos(2 * torch.pi * torch.sqrt(torch.sum(x ** 2))) + 0.1 * torch.sqrt(torch.sum(x ** 2))


class SchafferN2:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 0.5 + (torch.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2


class SchafferN4:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, 1.253115])]
        self.fglob = 0.292579
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x, y = xy[0], xy[1]
        return 0.5 + (torch.cos(torch.sin(torch.abs(x ** 2 - y ** 2))) ** 2 - 0.5) / (
                1 + 0.001 * (x ** 2 + y ** 2)) ** 2


class Schwefel:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.full((self.dim,), 420.968746)]
        self.fglob = 0
        self.bounds = [(-500, 500)] * self.dim

    def fun(self, x):
        return 418.982887 * self.dim - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


class Sphere:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-5.12, 5.12)] * self.dim

    def fun(self, x):
        return torch.sum(x ** 2)


class Step:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        return sum([torch.floor(torch.abs(v)) for v in xy])


class StyblinskiTang:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.full((self.dim,), -2.903534018185960)]
        self.fglob = -39.166172 * self.dim
        self.bounds = [(-5, 5)] * self.dim

    def fun(self, x):
        return torch.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2


class SumOfDifferentPowers:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-1, 1)] * self.dim

    def fun(self, x):
        return sum([torch.abs(item) ** (i + 2) for i, item in enumerate(x)])


class SumOfSquares:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim

    def fun(self, xy):
        return sum([i * v ** 2 for i, v in enumerate(xy)])


class Trid:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.tensor([i * (self.dim + 1 - i) for i in range(1, self.dim + 1)])]
        self.fglob = -self.dim * (self.dim + 4) * (self.dim - 1) / 6
        self.bounds = [(-self.dim ** 2, self.dim ** 2)] * self.dim

    def fun(self, x):
        a = sum([(x[i] - 1) ** 2 for i in range(self.dim)])
        b = sum([(x[i - 1] * x[i]) for i in range(1, self.dim)])
        return a - b


class Tripod:
    def __init__(self, dim):
        self.dim = 2
        self.N = self.dim
        self.global_optimum = [torch.tensor([0, -50])]
        self.fglob = 0
        self.bounds = [(-100, 100)] * self.dim

    def fun(self, xy):
        x1, x2 = xy[0], xy[1]
        p_x1 = 1 if x1 >= 0 else 0
        p_x2 = 1 if x2 >= 0 else 0

        a = p_x2 * (1 + p_x1)
        b = torch.abs(x1 + 50 * p_x2 * (1 - 2 * p_x1))
        c = torch.abs(x2 + 50 * (1 - 2 * p_x2))
        return a + b + c


class Wolfe:
    def __init__(self, dim=3):
        self.dim = 3
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(0, 2)] * self.dim

    def fun(self, xyz):
        x, y, z = xyz[0], xyz[1], xyz[2]
        return (4.0 / 3.0) * (x ** 2.0 + y ** 2.0 - x * y) ** 0.75 + z


class Zakharov:
    def __init__(self, dim):
        self.dim = dim
        self.N = self.dim
        self.global_optimum = [torch.zeros(self.dim)]
        self.fglob = 0
        self.bounds = [(-10, 10)] * self.dim
        self.log = True

    def fun(self, x):
        i = torch.arange(1, self.dim + 1)
        return torch.sum(x ** 2) + torch.sum((0.5 * i * x) ** 2) + torch.sum((0.5 * i * x) ** 4)


FUNCS = {
    'Ackley': Ackley,
    'AckleyN2': AckleyN2,
    'Adjiman': Adjiman,
    'AMGM': AMGM,
    'BartelsConn': BartelsConn,
    'Beale': Beale,
    'BentCigar': BentCigar,
    'Bird': Bird,
    'BohachevskyN1': BohachevskyN1,
    'BohachevskyN2': BohachevskyN2,
    'BohachevskyN3': BohachevskyN3,
    'Booth': Booth,
    'Branin': Branin,
    'Brent': Brent,
    'Brown': Brown,
    'BukinN6': BukinN6,
    'CamelHump3': CamelHump3,
    'CamelHump6': CamelHump6,
    'CarromTable': CarromTable,
    'Chichinadze': Chichinadze,
    'ChungReynolds': ChungReynolds,
    'Colville': Colville,
    'CosineMixture': CosineMixture,
    'Csendes': Csendes,
    'Cube': Cube,
    'Damavandi': Damavandi,
    'DeckkersAarts': DeckkersAarts,
    'DixonPrice': DixonPrice,
    'DropWave': DropWave,
    'Easom': Easom,
    'Eggholder': Eggholder,
    'Exponential': Exponential,
    'FreudensteinRoth': FreudensteinRoth,
    'GoldsteinPrice': GoldsteinPrice,
    'Griewank': Griewank,
    'Himmelblau': Himmelblau,
    'HolderTable': HolderTable,
    'Hosaki': Hosaki,
    'Keane': Keane,
    'Leon': Leon,
    'LeviN13': LeviN13,
    'Matyas': Matyas,
    'Michalewicz': Michalewicz,
    'McCormick': McCormick,
    'Parsopoulos': Parsopoulos,
    'PenHolder': PenHolder,
    'Qing': Qing,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'Salomon': Salomon,
    'SchafferN2': SchafferN2,
    'SchafferN4': SchafferN4,
    'Schwefel': Schwefel,
    'Sphere': Sphere,
    'Step': Step,
    'StyblinskiTang': StyblinskiTang,
    'SumOfDifferentPowers': SumOfDifferentPowers,
    'SumOfSquares': SumOfSquares,
    'Trid': Trid,
    'Tripod': Tripod,
    'Wolfe': Wolfe,
    'Zakharov': Zakharov,

}
