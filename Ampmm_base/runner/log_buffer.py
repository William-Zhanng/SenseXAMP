from collections import OrderedDict
import numpy as np

class LogBuffer:
    """
    Save intermediate results of runner.
    """
    def __init__(self):
        self.variables_history = OrderedDict()
        self.num_history = OrderedDict()
        self.output = OrderedDict()
        self.empty = True

    def clear_output(self):
        self.output.clear()
        self.empty = True

    def clear_content(self,key):
        if key in self.variables_history.keys():
            self.variables_history[key] = []
            self.num_history[key] = []

    def clear_all(self):
        self.variables_history.clear()
        self.num_history.clear()
        self.clear_output()
    
    def update(self, vars: dict, count: int=1):
        assert isinstance(vars, dict)
        assert count >= 1
        for key, var in vars.items():
            if key not in self.variables_history:
                self.variables_history[key] = []
                self.num_history[key] = []
            self.variables_history[key].append(var)
            self.num_history[key].append(count)
    
    def output_results(self, n):
        """
        Output average of latest n values 
        """
        assert n > 0
        for key in self.variables_history:
            values = np.array(self.variables_history[key][-n:])
            nums = np.array(self.num_history[key][-n:])
            if key == 'batch_time':
                self.output[key] = np.sum(values * nums)
            else:
                self.output[key] = np.sum(values * nums) / np.sum(nums)
        return self.output