import operator

class Meter():

    def __init__(self, resolve_func=None):
        self.n = 0
        self.sum = None
        self.resolve_func = resolve_func

    def add(self, value):
        self.n += 1
        if self.sum:
            if type(value) is tuple:
                self.sum = tuple(map(operator.add, self.sum, value))
            else:
                self.sum += value
        else:
            self.sum = value

    def value(self):
        if self.resolve_func:
            return self.resolve_func(self.sum)
        else:
            return self.sum / self.n
