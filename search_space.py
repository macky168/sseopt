import random

import numpy as np

SCALE = 0.15

class categorical:
    def __init__(self, candidates_lst):
        self.candidates_lst = candidates_lst
        
    def select(self):
        return random.choice(self.candidates_lst)
    
    def mutate(self, original_value):
        return random.choice(self.candidates_lst)
    
    def full_candidates_lst(self):
        return self.candidates_lst


class discrete:
    def __init__(self, min_value, max_value, step):
        if min_value > max_value:
            TypeError("Illegal search range")
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        
    def select(self):
        candidates_lst = []
    
        value = self.min_value
        while value <= self.max_value:
            candidates_lst.append(value)
            value += self.step
            
        return random.choice(candidates_lst)
    
    def mutate(self, original_value):
        """
        <example>
        
        min_value = 10, max_value = 16, step = 2
        original_value = 12
        ↓
        candidates_lst_len = 4
        rand = -0.26 (normal distribution)
        ↓ (by normal distribution)
        added value = -1
        ↓
        return 12 * (-1*2) = 10
        """
        candidates_lst_len = 0
        value = self.min_value
        while value <= self.max_value:
            candidates_lst_len += 1
            value += self.step
        
        rand = np.random.normal(loc=0.0, scale=SCALE, size=1).tolist()[0]
        added_value = round(rand * candidates_lst_len)
        
        if (self.min_value < float(added_value)*float(self.step) + original_value and 
            float(added_value)*float(self.step) + original_value < self.max_value):
            original_value += added_value*self.step
        else:
            # no mutation
            pass
        
        return original_value
    
    def full_candidates_lst(self):
        candidates_lst = []
    
        value = self.min_value
        while value <= self.max_value:
            candidates_lst.append(value)
            value += self.step
        
        return candidates_lst


class discrete_int:
    def __init__(self, min_value, max_value):
        if min_value > max_value:
            TypeError("Illegal search range")
        self.min_value = min_value
        self.max_value = max_value
        self.step = 1
        
    def select(self):
        candidates_lst = []
    
        value = self.min_value
        while value <= self.max_value:
            candidates_lst.append(value)
            value += self.step
            
        return random.choice(candidates_lst)
    
    def mutate(self, original_value):
        candidates_lst_len = 0
        value = self.min_value
        while value <= self.max_value:
            candidates_lst_len += 1
            value += self.step
        
        rand = np.random.normal(loc=0.0, scale=SCALE, size=1).tolist()[0]
        added_value = round(rand * candidates_lst_len)
            
        if (self.min_value < float(added_value)*float(self.step) + original_value and 
            float(added_value)*float(self.step) + original_value < self.max_value):
            original_value += added_value*self.step
        else:
            # no mutation
            pass
        
        return original_value
    
    def full_candidates_lst(self):
        candidates_lst = []
    
        value = self.min_value
        while value <= self.max_value:
            candidates_lst.append(value)
            value += self.step
            
        return candidates_lst
    
    
class fixed:
    def __init__(self, value):
        self.value = value
        
    def select(self):            
        return self.value
    
    def mutate(self, original_value):        
        return self.value
    
    def full_candidates_lst(self):
        return [self.value]
    