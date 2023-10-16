import csdl
from m3l.core.m3l_classes import Variable
from m3l.core.m3l_standard_operations import *


def add(x1, x2):
    '''
    Performs addition of two M3L variables, i.e., x1 + x2
    '''
    addition_operation = Add()

    return addition_operation.evaluate(x1, x2)

def subtract(x1, x2):
    """
    Performs subtraction of two M3L variabls, i.e., x1 - x2
    """
    subtraction_operation = Subtract()

    return subtraction_operation.evaluate(x1, x2)

def reshape(x : Variable, shape : tuple):
    """
    Performs reshaping of an m3l variable

    Parameters:
    ----------
    x : Variable
        The m3l variable to be reshaped
    shape : tuple
        The shape to which the m3l variable is to be reshaped
    """
    reshape_operation = Reshape(shape=shape)

    return reshape_operation.evaluate(x)

def norm(x : Variable, order:int=2, axes:tuple=(-1, )):
    """
    Performs p-norm of an m3l variable along specified axes.

    Parameters:
    ----------
    x : Variable
        The m3l variable on which the p-norm is to be performed
    order : int (default = 2)
        The order of the p-norm. 
    axes :  tuple (default = (-1, ))
        The axes along which the p-norm is to be performed 
    """
    norm_operation = Norm(order=order, axes=axes)

    return norm_operation.evaluate(x)

def cross(x1 : Variable, x2 : Variable, axis : int=0):
    """
    Performs cross product of two m3l variables
    """
    cross_product_operation = CrossProduct(axis=axis)

    return cross_product_operation.evaluate(x1=x1, x2=x2)


def multiply(x1 : Variable, x2 : Variable):
    """
    Performs multiplication of two m3l variables, i.e., x1 * x2
    """
    multiplication_operation = Multiplication()

    return multiplication_operation.evaluate(x1=x1, x2=x2)

def divide(x1 : Variable, x2 : Variable):
    """
    Performs division of two m3l variables, i.e., x1 / x2
    """
    if type(x1) is int or type(x1) is float:
        var1 = Variable('x1', shape=(1, ), value=x1)
    else:
        var1 = x1
    if type(x2) is int or type(x2) is float:
        var2 = Variable('x2', shape=(1, ), value=x2)
    else:
        var2 = x2
    division_operation = Division()

    return division_operation.evaluate(x1=var1, x2=var2)

def vstack(x1 : Variable, x2: Variable):
    """
    Performs vertical stacking of two m3l variables
    """

    vstack_operation = VStack()

    return vstack_operation.evaluate(x1=x1, x2=x2)


def linear_combination(start : Variable, stop : Variable, num_steps:int=50, 
                       start_weights:np.ndarray=None, stop_weights:np.ndarray=None) -> Variable:
    """
    Performs a linear combination of two m3l variables. The linear combination is defined as:

    Parameters:
    ----------
    start : Variable
        The starting m3l variable
    stop : Variable
        The stopping m3l variable
    num_steps : int (default = 50)
        The number of steps in the linear combination
    start_weights : np.ndarray (default = None)
        The weights for the starting m3l variable
    stop_weights : np.ndarray (default = None)
        The weights for the stopping m3l variable
    """
    if num_steps is not None and start_weights is None and stop_weights is None:
            linspace(start, stop, num_steps)

    if len(start.shape) == 1:
        num_per_step = start.shape[0]
    else:
        num_per_step = np.prod(start.shape)

    map_num_outputs = num_steps*num_per_step
    map_num_inputs = num_per_step
    map_start = sps.lil_matrix((map_num_outputs, map_num_inputs))
    map_stop = sps.lil_matrix((map_num_outputs, map_num_inputs))
    for i in range(num_steps):
        start_step_map = (sps.eye(num_per_step)) * start_weights[i]
        map_start[i*num_per_step:(i+1)*num_per_step, :] = start_step_map

        stop_step_map = (sps.eye(num_per_step)) * stop_weights[i]
        map_stop[i*num_per_step:(i+1)*num_per_step, :] = stop_step_map

    map_start = map_start.tocsc()
    map_stop = map_stop.tocsc()

    flattened_start = start.reshape((num_per_step,))
    flattened_stop = stop.reshape((num_per_step,))
    mapped_start_array = matvec(map_start, flattened_start)
    mapped_stop_array = matvec(map_stop, flattened_stop)

    flattened_output = mapped_start_array + mapped_stop_array

    output = flattened_output.reshape((num_steps,) + tuple(start.shape))
    output.name = f'{start.name}_to_{stop.name}_linear_combination'

    return output


def linspace(start:Variable, stop:Variable, num_steps:int=50) -> Variable:
    """
    Performs a linear combination of two m3l variables. The linear combination is defined as:

    Parameters:
    ----------
    start : Variable
        The starting m3l variable
    stop : Variable
        The stopping m3l variable
    num_steps : int (default = 50)
        The number of steps in the linear combination
    """
    if num_steps == 1:
        stop_weights = np.array([0.5])
    else:
        stop_weights = np.arange(num_steps)/(num_steps-1)
    start_weights = 1 - stop_weights

    return linear_combination(start=start, stop=stop, num_steps=num_steps, 
                              start_weights=start_weights, stop_weights=stop_weights)


def matvec(map : Variable, x : Variable):
    """
    Performs matrix-vector multiplication of two m3l variables
    """
    matvec_operation = MatVec()

    return matvec_operation.evaluate(map=map, x=x)