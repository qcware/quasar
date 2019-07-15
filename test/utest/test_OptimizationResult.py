import quasar
import numpy as np

"""
Test "OptimizationResult" Class
"""

def util_get_optimization():
    """
    A utility function for the testing functions below.
    """
    ket1 = quasar.Ket.from_int(9,5)
    ket2 = quasar.Ket.from_int(19,5)
    ket3 = quasar.Ket.from_int(29,5)
    o_result = quasar.OptimizationResult({ket1:(0.51,101), ket2:(0.42,202), ket3:(0.63,303)})
    return o_result


def initialization():
    """
    Validate the initialization (__init__()) of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()

    return [x for x in o_result.items()]==[('01001', (0.51,101)), ('10011', (0.42,202)), ('11101', (0.63,303))]


def contains():
    """
    Validate the __contains__() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()
    if not '01001' in o_result:
        return False
    
    return True
        
    
def getitem():
    """
    Validate the __getitem__() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()

    return o_result['10011']==(0.42,202)
    
    
def setitem():
    """
    Validate the __setitem__() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()
    o_result['11111']=(0.99, 505)

    return o_result['11111']==(0.99, 505)    
    

def get():
    """
    Validate the get() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()

    return o_result.get('10011')==(0.42,202)    
    
    
def setdefault():
    """
    Validate the get() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()
    o_result.setdefault('10011', (0.88, 8))
    o_result.setdefault('11111', (0.88, 8))

    return o_result.get('10011')==(0.42,202) and o_result.get('11111')==(0.88, 8)   
    
    
def N():
    """
    Validate the N() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()

    return o_result.N==5
      

def str():
    """
    Validate the __str__() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()
    
    return o_result.__str__().replace('\n', '')=='|10011> :   0.420000 202|01001> :   0.510000 101|11101> :   0.630000 303'      

    
def energy_sorted():
    """
    Validate the energy_sorted() of the "OptimizationResult" class.
    """
    o_result = util_get_optimization()
    o_result = o_result.energy_sorted

    return list(o_result.values())[0][0]==0.42

    
def merge():
    """
    Validate the merge() of the "OptimizationResult" class.
    """
    ket1 = quasar.Ket.from_int(9,5)
    ket2 = quasar.Ket.from_int(19,5)
    ket3 = quasar.Ket.from_int(29,5)
    o_result1 = quasar.OptimizationResult({ket1:(0.51,101), ket2:(0.42,202)})
    o_result2 = quasar.OptimizationResult({ket3:(0.63,303)})
    
    o_result = quasar.OptimizationResult.merge([o_result1, o_result2])
    
    return [x for x in o_result.items()]==[('01001', (0.51,101)), ('10011', (0.42,202)), ('11101', (0.63,303))]    
   
    
    
