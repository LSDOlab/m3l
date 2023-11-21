import csdl
from m3l.core.m3l_classes import Variable
from m3l.core.m3l_standard_operations import *
from typing import Union

def copy(x : Variable):
    """
    Performs a deep copy of an m3l variable
    """
    copy_var = Variable(name=x.name, shape=x.shape, operation=x.operation, value=x.value,
                        dv_flag=x.dv_flag, lower=x.lower, upper=x.upper, scaler=x.scaler,
                        equals=x.equals)
    return copy_var

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

def expand(x : Variable, new_shape, indices : str = None):
    """
    Performs expansion of an m3l variable
    ----------
    x : Variable
        The m3l variable to be expanded
    new_shape : tuple
        The shape to which the m3l variable is to be expanded to
    """

    expand_operation = Expand(new_shape=new_shape, indices=indices)

    return expand_operation.evaluate(x)

def dot(x1 : Variable, x2: Variable, axis : int = None):
    """
    Performs dot product of 2 m3l variables
    ----------
    x1 : Variable
        the first m3l variable
    x2 : Variable
        the second m3l variable
    axis : int
        The axis across which to perform the dot product
    """

    dot_operation = Dot()

    return dot_operation.evaluate(x1=x1, x2=x2, axis=axis)

def cos(x : Variable):
    """
    Performs cosine operation of an m3l variable
    """

    cos_operation = Cos()

    return cos_operation.evaluate(x=x)


def sin(x : Variable):
    """
    Performs sine operation of an m3l variable
    """

    sin_operation = Sin()

    return sin_operation.evaluate(x=x)

def arccos(x : Variable):
    """
    Performs inverse cosine operation of an m3l variable
    """

    arccos_operation = ArcCos()

    return arccos_operation.evaluate(x=x)


def arcsin(x : Variable):
    """
    Performs inverse sine operation of an m3l variable
    """

    arcsin_operation = ArcSin()

    return arcsin_operation.evaluate(x=x)


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

def power(x1 : Union[Variable, int, float], x2: Union[Variable, float, int]):
    """
    Performs power operation between m3l variable or an m3l variable and float/int
    """
    power_operation = Power()

    return power_operation.evaluate(x1=x1, x2=x2)


def multiply(x1 : Union[Variable, int, float], x2 : Variable):
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
        var2 = Variable(name='x2', shape=(1, ), value=x2)
    else:
        var2 = x2
    division_operation = Division()

    return division_operation.evaluate(x1=var1, x2=var2)

def sum(x:Variable, axes:tuple):
    """
    Performs summation of an m3l variable along specified axes.

    Parameters:
    ----------
    x : Variable
        The m3l variable on which the summation is to be performed
    axes :  tuple
        The axes along which the summation is to be performed 
    """
    sum_operation = Sum(axes=axes)

    return sum_operation.evaluate(x)

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
    # output.name = f'{start.name}_to_{stop.name}_linear_combination'

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


def matvec(map : sps.csc_matrix, x : Variable):
    """
    Performs matrix-vector multiplication of two m3l variables
    """
    matvec_operation = MatVec(map=map)

    return matvec_operation.evaluate(x=x)

def matmat(map : Variable, x : Variable):
    """
    Performs matrix-vector multiplication of two m3l variables
    """
    matmat_operation = MatMat()

    return matmat_operation.evaluate(map=map, x=x)


def rotate(points:Variable, axis_origin:Variable, axis_vector:Variable, angles:Variable, units:str='degrees'):
    """
    Performs rotation of an m3l variable about an axis by an angle
    """
    rotation_operation = Rotate(units=units)

    return rotation_operation.evaluate(points=points, axis_origin=axis_origin, axis_vector=axis_vector, angles=angles)


def variable_get_item(x:Variable, indices:np.ndarray):
    """
    Performs indexing of an m3l variable
    """
    # original_shape = x.shape
    # if len(x.shape) > 1:
    #     x_flat = x.reshape((np.prod(x.shape),))
    # else:
    #     x_flat = x

    map_num_outputs = indices.shape[0]
    map_num_inputs = x.shape[0]
    map = sps.lil_matrix((map_num_outputs, map_num_inputs))
    for i in range(map_num_outputs):
        map[i, indices[i]] = 1

    map = map.tocsc()

    if len(x.shape) == 1:
        indexed_x = matvec(map=map, x=x.copy())
    else:
        indexed_x = matmat(map=map, x=x.copy())

    # if len(x.shape) > 1:
    #     index_x = index_x_flat.reshape(original_shape)
    # else:
    #     index_x = index_x_flat

    return indexed_x


def variable_set_item(x:Variable, indices:np.ndarray, value:Variable):
    """
    Performs indexing/assignment of an m3l variable
    """
    # original_shape = x.shape
    # if len(x.shape) > 1:
    #     x_flat = x.reshape((np.prod(x.shape),))
    # else:
    #     x_flat = x

    import m3l

    # updated component
    map_num_outputs = x.shape[0]
    map_num_inputs = indices.shape[0]
    map = sps.lil_matrix((map_num_outputs, map_num_inputs))
    for i in range(indices.shape[0]):
        index = indices[i]
        map[index, i] = 1
    map = map.tocsc()
    x_updated = matvec(map=map, x=value)

    # unchanged component
    data = np.ones((x.shape[0] - indices.shape[0],))
    unchanged_indices = np.delete(np.arange(x.shape[0]), indices)
    unchanged_indexing_map = sps.coo_matrix((data, (unchanged_indices, unchanged_indices)),
                                shape=(x.shape[0], x.shape[0]))
    unchanged_indexing_map = unchanged_indexing_map.tocsc()
    # x_unchanged = matvec(map=unchanged_indexing_map, x=x)
    x_unchanged = matvec(map=unchanged_indexing_map, x=x.copy())

    new_x = x_updated + x_unchanged

    # if len(x.shape) > 1:
    #     index_x = index_x_flat.reshape(original_shape)
    # else:
    #     index_x = index_x_flat

    return new_x