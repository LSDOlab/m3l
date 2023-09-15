import csdl
import m3l


def add(x1, x2):
    '''
    Performs addition with between two M3L variables.
    '''

    import m3l
    addition_operation = m3l.Add()

    return addition_operation.evaluate(x1, x2)


def create_input(name : str, ) -> m3l.Variable:
    """
    Create an M3L variable.

    Parameters:
    ----------
        name : str 
            The name of the variable 
        

    """
    
    return