import csdl

def add(x1, x2):
    '''
    Performs addition with between two M3L variables.
    '''

    import m3l
    addition_operation = m3l.Add()

    return addition_operation.evaluate(x1, x2)
