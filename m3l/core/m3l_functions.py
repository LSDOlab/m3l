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

def norm(x : Variable, order:int=2, axes:tuple=(-1, )):
    """
    Performs p-norm of an m3l variabke along specified axes.

    Parameters:
    ----------
    x
    
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
    division_operation = Division()

    return division_operation.evaluate(x1=x1, x2=x2)

def vstack(x1 : Variable, x2: Variable):
    """
    Performs vertical stacking of two m3l variables
    """

    vstack_operation = VStack()

    return vstack_operation.evaluate(x1=x1, x2=x2)