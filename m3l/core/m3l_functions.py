import csdl

def add(x1, x2):
    '''
    Performs addition with between two M3L variables.
    '''
    # x1_name = x1.name
    # x2_name = x2.name
    # if x2 == x1:
    #     x2_name = f'{x2_name}_'

    operation_csdl = csdl.Model()
    x1_csdl = operation_csdl.declare_variable(name=x1.name, shape=x1.shape)
    x2_csdl = operation_csdl.declare_variable(name=x2.name, shape=x2.shape)
    y = x1_csdl + x2_csdl
    output_name = f'{x1.name}_plus_{x2.name}'
    operation_csdl.register_output(name=output_name, var=y)

    import m3l
    arguments = {x1.name:x1, x2.name:x2}
    csdl_operation = m3l.CSDLOperation(name=f'{output_name}_operation', arguments=arguments, operation_csdl=operation_csdl)
    return m3l.Variable(name=output_name, shape=y.shape, operation=csdl_operation)
