import numpy as np


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self, resolve_func=None):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.resolve_func = resolve_func

    def add(self, value, n=1):
        self.n += n

        if self.n == 0:
            self.sum = np.nan
        elif self.n == 1:
            self.sum = n*value
        else:
            self.sum += n*value

    def value(self):
        if self.resolve_func is None:
            if self.n == 0:
                return np.nan
            else:
                return self.sum / self.n
        else:
            return self.resolve_func(self.sum)

    def reset(self):
        self.n = 0
        self.sum = np.nan
