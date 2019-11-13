import math
import random

class UncorrNoiseConso:
    def __init__(self, muconso=None, sigmaConso=None):
        self.e = math.exp(1)
        self.need_init_ = sigmaConso is None or muconso is None  # need to be initialized with the variation of loads
        self.sigmaConso = sigmaConso
        self.muconso = muconso

    def need_init(self):
        return self.need_init_

    def init(self, muconso, sigmaConso):
        self.muconso = muconso
        self.sigmaConso = sigmaConso
        self.need_init_ = False

    def __call__(self, loads):
        """Compute the 'uncorrelated noise' for a single load"""
        if self.need_init_:
            msg = "Trying to use an un initialize uncorrelated noise"
            raise RuntimeError(msg)

        loads = [self.recoeverything(el, self.muconso) for el in loads]
        return [load*random.lognormvariate(mu=1., sigma=self.sigmaConso)/self.e for load in loads]
        # return load*random.lognormvariate(mu=1., sigma=self.sigmaConso)/self.e

    def recoeverything(self, el, avg):
        """reconnect everything
        If something was disconnected, its new value is drawn to be +- 50% of the avg value"""
        res = el
        if res == 0.:
            res = avg*random.lognormvariate(mu=1., sigma=0.5)/self.e
        return res
