import numpy as np
from m3l.core.m3l_classes import *

def index_functions(names:list, property_name:str, space:FunctionSpace, length:int, value:np.ndarray=None):
    space_dict = {}
    coeff_dict = {}
    for name in names:
        space_dict[name] = space
        if value is None:
            coeff = Variable(name = name+'_'+property_name+'_coefficients', shape = space.coefficients_shape + (length,))
        else:
            coeff = Variable(name = name+'_'+property_name+'_coefficients', shape = space.coefficients_shape + (length,), value=value)
        coeff_dict[name] = coeff
    function_space = IndexedFunctionSpace(property_name + 'space', spaces=space_dict)
    return IndexedFunction(property_name + '_function', space=function_space, coefficients=coeff_dict)




