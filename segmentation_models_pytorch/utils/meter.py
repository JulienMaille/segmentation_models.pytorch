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
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.n += n

        if self.n == 0:
            self.sum = np.nan
        elif self.n == 1:
            self.sum = n*value
        else:
            self.sum += n*value

    def value(self):
        if isinstance(self.sum, np.ndarray):
            if self.sum[1] == 0:
                return np.nan
            else:
                return self.sum[0]/self.sum[1]
        else:
            if self.n == 0:
                return np.nan
            else:
                return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = np.nan
