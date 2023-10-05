import csdl
from m3l.core.m3l_classes import ExplicitOperation, Variable
import numpy as np
from m3l.utils.utility_functions import replace_periods_with_underscores
from python_csdl_backend import Simulator


class Norm(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('order', types=int, default=2)
        self.parameters.declare('axes', types=tuple, default=(-1, ))
    
    def assign_attributes(self):
        self.order = self.parameters['order']
        self.axes = self.parameters['axes']

    def compute(self):
        order = self.order
        axes = self.axes
        x = self.arguments['x']
        
        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(name='x', shape=x.shape)
        y = csdl.pnorm(x_csdl, pnorm_type=order, axis=axes)
        output_name = replace_periods_with_underscores(f'{x.name}_norm')
        csdl_model.register_output(name=output_name, var=y)
        return csdl_model

    def evaluate(self, x : Variable) -> Variable:
        self.arguments = {'x' : x}
        axes = self.parameters['axes']
        x_shape = x.shape

        self.name = f"norm_operation_{x.name}"

        new_axes = []
        for axis in axes:
            if axis < 0:
                new_axes.append(len(x_shape)+axis)
            else:
                new_axes.append(axis)
        
        new_axes = tuple(new_axes)

        out_shape = []
        for i, size in enumerate(x_shape):
            if i not in new_axes:
                out_shape.append(size)
        out_shape = tuple(out_shape)
        
        output_name = replace_periods_with_underscores(f'{x.name}_norm')
        norm = Variable(name=output_name, shape=out_shape, operation=self)

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x'] = x.value
        sim.run()
        norm.value = sim[output_name]

        return norm


    def compute_derivates(self): # Really this is compute 2nd derivatives 
        norm_derivative_model = csdl.Model()
        x_arg = self.arguments['x']
        p = self.order
        axes = self.axes
        x = norm_derivative_model.declare_variable(f'{x_arg.name}', shape=x_arg.shape)

        dx_norm_dx = x * ((x**2)**0.5)**(p-2) / csdl.pnorm(x, pnorm_type=p, axis=axes)**(p-1)
        norm_derivative_model.register_output(f'{x_arg.name}_norm_derivative', dx_norm_dx)




class Subtract(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('some_input', types=int, default=0)

    def assign_attributes(self):
        self.some_parameter = self.parameters['some_input']

    def compute(self):
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name='x2', shape=x2.shape)

        y = x1_csdl - x2_csdl
        output_name = replace_periods_with_underscores(f'{x1.name}_minus_{x2.name}')
        csdl_model.register_output(name=output_name, var=y)
        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable):
        self.name = f'{x1.name}_minus_{x2.name}_operation'
        self.arguments = {}
        self.arguments['x1'] = x1
        self.arguments['x2'] = x2

        output_name = replace_periods_with_underscores(f'{x1.name}_minus_{x2.name}')
        function_values = Variable(name=output_name, shape=x1.shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]

        
        return function_values

    def compute_derivates(self):
        return super().compute_derivates()

class Add(ExplicitOperation):

    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='addition_operation')

    def compute(self):
        '''
        Creates the CSDL model to compute the addition function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model}
            The csdl model or module that computes the model/operation outputs.
        '''
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name='x2', shape=x2.shape)

        y = x1_csdl + x2_csdl
        output_name = replace_periods_with_underscores(f'{x1.name}_plus_{x2.name}')
        # output_name = f'x2_plus_x1'
        csdl_model.register_output(name=output_name, var=y)
        return csdl_model

    def compute_derivates(self):
        '''
        -- optional --
        Creates the CSDL model to compute the derivatives of the model outputs. This is only needed for dynamic analysis.
        For now, I would recommend coming back to this.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the derivatives of the model/operation outputs.
        '''
        pass

    def evaluate(self, x1:Variable, x2:Variable) -> Variable:
        '''
        User-facing method that the user will call to define a model evaluation.

        Parameters
        ----------
        mesh : Variable
            The mesh over which the function will be evaluated.

        Returns
        -------
        function_values : Variable
            The values of the function at the mesh locations.
        '''
        self.name = f'{x1.name}_plus_{x2.name}_operation'
        self.parameters['name'] = self.name
        # self.x1 = x1
        # self.x2 = x2

        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}
    
        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{x1.name}_plus_{x2.name}')
        function_values = Variable(name=output_name, shape=x1.shape, operation=self)
    
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]


        
        return function_values


class Multiplication(ExplicitOperation):
    """
    Multiplcation class. Subclass of M3Ls ExplicitOperation 
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='multiplication_operation')

    def compute(self):
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name='x2', shape=x2.shape)

        y = x1_csdl * x2_csdl
        output_name = replace_periods_with_underscores(f'{x1.name}_times_{x2.name}')
        csdl_model.register_output(name=output_name, var=y)
        return csdl_model
    
    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        self.name = f'{x1.name}_multiplication_{x2.name}_operation'
        self.parameters['name'] = self.name
        
        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}

        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{x1.name}_times_{x2.name}')
        function_values = Variable(name=output_name, shape=x1.shape, operation=self)

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]

        return function_values

class Division(ExplicitOperation):
    """
    Division class. Subclass of M3Ls ExplicitOperation 
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='division_operation')


    def compute(self):
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name='x2', shape=x2.shape)

        y = x1_csdl / x2_csdl
        output_name = replace_periods_with_underscores( f'{x1.name}_divide_by_{x2.name}')
        csdl_model.register_output(name=output_name, var=y)
        return csdl_model
    
    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        self.name = f'{x1.name}_division_{x2.name}_operation'
        
        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}

        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{x1.name}_divide_by_{x2.name}')
        function_values = Variable(name=output_name, shape=x1.shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]
        return function_values

class CrossProduct(ExplicitOperation):
    """
    Cross product class to perform the cross product of two m3l variables
    """
    def initialize(self, kwargs):
        self.parameters.declare('axis', types=int)

    def assign_attributes(self):
        self.axis = self.parameters['axis']

    def compute(self):
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name='x2', shape=x2.shape)

        y = csdl.cross(x1_csdl, x2_csdl, axis=self.axis)
        output_name = replace_periods_with_underscores(f'{x1.name}_cross_{x2.name}')
        csdl_model.register_output(name=output_name, var=y)

        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        self.name = f"{x1.name}_cross_{x2.name}_operation"
        self.arguments = {'x1': x1, 'x2' : x2}

        output_name = replace_periods_with_underscores(f'{x1.name}_cross_{x2.name}')
        function_values = Variable(name=output_name, shape=x1.shape, operation=self)

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]

        
        return function_values



class VStack(ExplicitOperation):
    """
    v-stack class for stacking two m3l variables vertically
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='vstack_operation')
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']
        # shape = x1.shape
        # shape[0] = x2.shape[0]
        shape = self.shape
        output_name = replace_periods_with_underscores(f'{x1.name}_stack_{x2.name}')
        operation_csdl = csdl.Model()
        x1_csdl = operation_csdl.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = operation_csdl.declare_variable(name='x2', shape=x2.shape)
        y = operation_csdl.create_output(name=output_name, shape=shape)
        y[0:x1.shape[0],:] = x1_csdl
        y[x1.shape[0]:,:] = x2_csdl
        # operation_csdl.register_output(name=output_name, var=y)
        return operation_csdl

    def compute_derivates(self):
        '''
        -- optional --
        Creates the CSDL model to compute the derivatives of the model outputs. This is only needed for dynamic analysis.
        For now, I would recommend coming back to this.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the derivatives of the model/operation outputs.
        '''
        pass

    def evaluate(self, x1:Variable, x2:Variable) -> Variable:
        '''
        User-facing method that the user will call to define a model evaluation.

        Parameters
        ----------
        mesh : Variable
            The mesh over which the function will be evaluated.

        Returns
        -------
        function_values : Variable
            The values of the function at the mesh locations.
        '''
        self.name = f'{x1.name}_stack_{x2.name}_operation'

        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}
        # shape = x1.shape
        # shape[0] = x2.shape[0]

        self.shape = (x1.shape[0] + x2.shape[0], ) + x1.shape[1:]
        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{x1.name}_stack_{x2.name}')
        function_values = Variable(name=f'{x1.name}_stack_{x2.name}', shape=self.shape, operation=self)
        
        
         # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        function_values.value = sim[output_name]
        
        
        return function_values

