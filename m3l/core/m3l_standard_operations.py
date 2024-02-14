import csdl
from m3l.core.m3l_classes import ExplicitOperation, Variable
import numpy as np
import scipy.sparse as sps
from m3l.utils.utility_functions import replace_periods_with_underscores, generate_random_string
from python_csdl_backend import Simulator
import gc
from typing import Tuple


class Norm(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='norm_operation', types=str)
        self.parameters.declare('name', default='norm_operation', types=str)
        self.parameters.declare('order', types=int, default=2)
        self.parameters.declare('axes', types=tuple, default=(-1, ))
    
    def assign_attributes(self):
        self.order = self.parameters['order']
        self.axes = self.parameters['axes']
        self.name = self.parameters['name']

    def compute(self):
        order = self.order
        axes = self.axes
        x = self.arguments[f'x']
        
        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(name=f'x', shape=x.shape)
        if len(x.shape) == len(axes):
            y = csdl.pnorm(x_csdl, pnorm_type=order)
        else:
            y = csdl.pnorm(x_csdl, pnorm_type=order, axis=axes)
        csdl_model.register_output(name=self.output_name, var=y)
        self.csdl_model = csdl_model
        self.sim = Simulator(csdl_model)
        return csdl_model
    
    def compute_derivatives(self): # Really this is compute 2nd derivatives 
        norm_derivative_model = csdl.Model()
        x_arg = self.arguments['x']
        p = self.order
        axes = self.axes
        x = norm_derivative_model.declare_variable(f'x', shape=x_arg.shape)

        if len(x.shape) == len(axes):
            y = csdl.pnorm(x, pnorm_type=p)
        else:
            y = csdl.pnorm(x, pnorm_type=p, axis=axes)
        y_expanded = csdl.expand(y, shape=x.shape, indices='i->ij')

        # dx_norm_dx = x * ((x**2)**0.5)**(p-2) / y_expanded**(p-1)
        dx_norm_dx = x/y_expanded
        norm_derivative_model.register_output('d'+self.output_name+'_d'+'x', dx_norm_dx)
        self.derivate_csdl_model = norm_derivative_model
        self.derivative_sim = Simulator(norm_derivative_model)
        return norm_derivative_model

    def evaluate(self, x : Variable) -> Variable:
        axes = self.parameters['axes']
        x_shape = x.shape

        random_name = generate_random_string()
        self.name = f"norm_operation_{x.name}_{random_name}"
        self.arguments = {f'x' : x}

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
        
        if len(out_shape) == 0:
            out_shape = (1, )
        norm = Variable(shape=out_shape, operation=self)
        self.output_name = norm.name


        # create csdl model for in-line evaluations
        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x'] = x.value
            sim.run()
            norm.value = sim[self.output_name]
            # del operation_csdl
            del sim
            # gc.collect()
        return norm


class Cos(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='cos_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        x = self.arguments['x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable('x', shape=x.shape)

        cos_output = csdl.cos(x_csdl)

        csdl_model.register_output(self.output_name, cos_output)

        return csdl_model

    def evaluate(self, x : Variable):
        self.name = f"{x.name}_cos_operation"
        
        self.arguments = {
            f'x' : x,
        }

        output = Variable(shape=x.shape, operation=self)
        self.output_name = output.name
        

        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim['x'] = x.value
            sim.run()
            output.value = sim[self.output_name]
            # del operation_csdl
            del sim
            # gc.collect()
        return output


class Sin(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='sin_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']
    
    def compute(self):
        x = self.arguments['x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable('x', shape=x.shape)

        sin_output = csdl.sin(x_csdl)

        csdl_model.register_output(self.output_name, sin_output)
        return csdl_model

    def evaluate(self, x : Variable):
        self.name = f"{x.name}_sin_operation"
        self.arguments = {
            'x' : x,
        }

        output = Variable(shape=x.shape, operation=self)
        self.output_name = output.name

        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim['x'] = x.value
            sim.run()
            output.value = sim[self.output_name]

        return output
    
class ArcCos(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='arccos_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        x = self.arguments[f'x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(f'x', shape=x.shape)

        arccos_output = csdl.arccos(x_csdl)

        csdl_model.register_output(self.output_name, arccos_output)
        return csdl_model

    def evaluate(self, x : Variable):
        self.name = f"{x.name}_arccos_operation"
        

        output = Variable(shape=x.shape, operation=self)
        self.output_name = output.name
        self.arguments = {
            f'x' : x,
        }

        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x'] = x.value
            sim.run()
            output.value = sim[self.output_name]

        return output


class ArcSin(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='arcsin_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']
    
    def compute(self):
        x = self.arguments['x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable('x', shape=x.shape)

        arcsin_output = csdl.arcsin(x_csdl)

        csdl_model.register_output(self.output_name, arcsin_output)
        return csdl_model

    def evaluate(self, x : Variable):
        self.name = f"{x.name}_arcsin_operation"
        self.arguments = {
            'x' : x,
        }

        output = Variable(shape=x.shape, operation=self)
        self.output_name = output.name

        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim['x'] = x.value
            sim.run()
            output.value = sim[self.output_name]

        return output


class Dot(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='dot_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        x1 = self.arguments[f'x1']
        x2 = self.arguments[f'x2']

        axis = self.axis

        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(f'x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(f'x2', shape=x2.shape)

        dot = csdl.dot(x1_csdl, x2_csdl, axis=axis)
        csdl_model.register_output(self.output_name, dot)

        return csdl_model
    
    def compute_derivates(self):
        # TODO: Come back and implement this!
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']

        pass
    
    def evaluate(self, x1 : Variable, x2 : Variable, axis : int):
        self.name = f"{x1.name}_dot_{x2.name}_operation"
        self.axis = axis

        if axis is not None:
            shape_list = list(x1.shape)
            shape_list.pop(axis)
            new_shape = tuple(shape_list)
        else:
            new_shape = (1, )

        output = Variable(shape=new_shape, operation=self)
        self.output_name = output.name
        
        self.arguments = {
            f'x1' : x1,
            f'x2' : x2,
        }
        
        if (x1.value is not None) and (x2.value is not None):
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x1'] = x1.value
            sim[f'x2'] = x2.value
            sim.run()
            output.value = sim[self.output_name]

            del sim
            # del operation_csdl
            # gc.collect()

        return output


class Expand(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('new_shape', types=tuple)
        self.parameters.declare('indices', types=str, allow_none=True)

    def assign_attributes(self):
        self.new_shape = self.parameters['new_shape']
        self.indices = self.parameters['indices']

    def compute(self):
        x = self.arguments[f'x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(f'x', shape=x.shape)
        x_csdl_expanded = csdl.expand(x_csdl, shape=self.new_shape, indices=self.indices)

        csdl_model.register_output(self.output_name, x_csdl_expanded)

        return csdl_model

    def evaluate(self, x : Variable):
        random_name = generate_random_string()
        self.name = f'{x.name}_expand_operation_{random_name}'


        output = Variable(shape=self.new_shape, operation=self)
        self.output_name = output.name
        self.arguments = {f'x' : x}


        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x'] = x.value
            sim.run()
            output.value = sim[self.output_name]
            del sim
            # del operation_csdl 
            # gc.collect()
        return output


class Power(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='to_the_power_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        scalers = self.scalers
        arguments = self.arguments

        x1 = arguments[f'x1']
        x2 = scalers[f'x2']
        csdl_model = csdl.Model()
        x1_csdl = csdl_model.declare_variable(f'x1', shape=x1.shape)
        y = x1_csdl**x2

        csdl_model.register_output(self.output_name, y)

        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable):
        random_name = generate_random_string()
        self.name = f'{x1.name}_to_the_power_operation_{random_name}'

        if not isinstance(x1, Variable):
            raise ValueError(f"Base of exponenent operation has to be an m3l.Variable. Received type {type(x1)}")
        elif isinstance(x2, Variable):
            raise ValueError(f"Cannot raise an m3l variable to the power of another m3l variable yet")
        
        else:
            self.scalers = {}
            self.arguments = {}
            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            
            self.scalers[f'x2'] = x2
            self.arguments[f'x1'] = x1
            
            

            if x1.value is not None:
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x1'] = x1.value
                sim.run()
                output.value = sim[self.output_name]
                del sim
                # del operation_csdl
                # gc.collect()
            return output

class Copy(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='subtraction_operation')

    def assign_attributes(self):
        random_name = generate_random_string(3)
        self.name = f"{self.parameters['name']}_{random_name}"

    def compute(self):
        arguments = self.arguments

        x = arguments[f'x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(f'x', shape=x.shape)

        csdl_model.register_output(name=self.output_name, var=x_csdl*1)

        return csdl_model

    def evaluate(self, x : Variable):
        self.arguments = {}
        output = Variable(shape=x.shape, operation=self)
        self.output_name = output.name
        self.arguments[f'x'] = x

        if x.value is not None:
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x'] = x.value
            sim.run()
            output.value = sim[self.output_name]
            del sim
            # del operation_csdl
            # gc.collect()
        return output



class Subtract(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='subtraction_operation', types=str)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        scalers = self.scalers
        arguments = self.arguments

        if f'x1' in scalers:
            x1 = scalers[f'x1']
            x2 = arguments[f'x2']
            csdl_model = csdl.Model()
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)
            y = x1 - x2_csdl

            csdl_model.register_output(name=self.output_name, var=y)

        elif f'x2' in scalers:
            x1 = arguments[f'x1']
            x2 = scalers[f'x2']
            csdl_model = csdl.Model()
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            y = x1_csdl - x2

            csdl_model.register_output(name=self.output_name, var=y)
        else:
            # x1 = self.arguments[f'{self.output_name}_x1']
            # x2 = self.arguments[f'{self.output_name}_x2']
            x1 = self.arguments[f'x1']
            x2 = self.arguments[f'x2']

            csdl_model = csdl.Model()
            # x1_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x1', shape=x1.shape)
            # x2_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x2', shape=x2.shape)
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x1.shape)


            y = x1_csdl +  (-1*x2_csdl)
            csdl_model.register_output(name=self.output_name, var=y)
        
        # csdl_model.print_var(y)
        
        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable):
        random_name = generate_random_string()
        if isinstance(x1, (float, int, np.ndarray)):
            self.name = f'scaler_minus_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.scalers[f'x1'] = x1
            self.arguments[f'x2'] = x2
            

            output = Variable(shape=x2.shape, operation=self)
            self.output_name = output.name
            # self.scalers[f'x1'] = x1
            # self.arguments[f'x2'] = x2
            
            # NOTE: in-line evaluations only work if all solver developers implement them
            # create csdl model for in-line evaluations
            if x2.value is not None:
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x2'] = x2.value
                sim.run()
                output.value = sim[self.output_name]
                del sim
                # del operation_csdl
                # gc.collect()

        elif isinstance(x2, (float, int, np.ndarray)):
            self.name = f'{x1.name}_minus_scaler_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.scalers[f'x2'] = x2
            self.arguments[f'x1'] = x1

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            # self.scalers[f'x2'] = x2
            # self.arguments[f'x1'] = x1

            # create csdl model for in-line evaluations
            if x1.value is not None:
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x1'] = x1.value
                sim.run()
                output.value = sim[self.output_name]
                del sim
                # del operation_csdl
                # gc.collect()
        
        else:
            self.name = f'{x1.name}_minus_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.arguments[f'x1'] = x1
            self.arguments[f'x2'] = x2

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            # self.arguments[f'{self.output_name}_x1'] = x1
            # self.arguments[f'{self.output_name}_x2'] = x2

            # create csdl model for in-line evaluations
            if (x1.value is not None) and (x2.value is not None):
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                # sim[f'{self.output_name}_x1'] = x1.value
                # sim[f'{self.output_name}_x2'] = x2.value
                sim[f'x1'] = x1.value
                sim[f'x2'] = x2.value
                sim.run()
                output.value = sim[self.output_name]
                del sim
                # del operation_csdl
                # gc.collect()

        
        return output

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
        # x1 = self.arguments[f'{self.output_name}_x1']
        # x2 = self.arguments[f'{self.output_name}_x2']

        # csdl_model = csdl.Model()
        # x1_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x1', shape=x1.shape)
        # x2_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x2', shape=x2.shape)

        # y = x1_csdl + x2_csdl
        # csdl_model.register_output(name=self.output_name, var=y)
        # return csdl_model
    
        scalers = self.scalers
        arguments = self.arguments

        if f'x1' in scalers:
            x1 = scalers[f'x1']
            x2 = arguments[f'x2']
            csdl_model = csdl.Model()
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)
            y = x1 + x2_csdl

            csdl_model.register_output(name=self.output_name, var=y)

        elif f'x2' in scalers:
            x1 = arguments[f'x1']
            x2 = scalers[f'x2']
            csdl_model = csdl.Model()
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            y = x1_csdl + x2

            csdl_model.register_output(name=self.output_name, var=y)
        else:
            x1 = self.arguments[f'x1']
            x2 = self.arguments[f'x2']

            csdl_model = csdl.Model()
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)

            y = x1_csdl + x2_csdl
            csdl_model.register_output(name=self.output_name, var=y)
        
        # csdl_model.print_var(y)
        
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
        output : Variable
            The values of the function at the mesh locations.
        '''
        random_name = generate_random_string(5)
        self.name = f'addition_operation_{random_name}'
        self.parameters['name'] = self.name
        # self.x1 = x1
        # self.x2 = x
    
        if isinstance(x1, (float, int, np.ndarray)):
            self.name = f'scaler_plus_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            

            output = Variable(shape=x2.shape, operation=self)
            self.output_name = output.name
            self.scalers[f'x1'] = x1
            self.arguments[f'x2'] = x2
            
            # NOTE: in-line evaluations only work if all solver developers implement them
            # create csdl model for in-line evaluations
            if x2.value is not None:
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x2'] = x2.value
                sim.run()
                output.value = sim[self.output_name]

        elif isinstance(x2, (float, int, np.ndarray)):
            self.name = f'{x1.name}_plus_scaler_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            self.scalers[f'x2'] = x2
            self.arguments[f'x1'] = x1

            # create csdl model for in-line evaluations
            if x1.value is not None:
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x1'] = x1.value
                sim.run()
                output.value = sim[self.output_name]
        
        else:
            self.name = f'{x1.name}_plus_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            self.arguments[f'x1'] = x1
            self.arguments[f'x2'] = x2

            # create csdl model for in-line evaluations
            if (x1.value is not None) and (x2.value is not None):
                operation_csdl = self.compute()
                sim = Simulator(operation_csdl)
                sim[f'x1'] = x1.value
                sim[f'x2'] = x2.value
                sim.run()
                output.value = sim[self.output_name]
        
        return output



class Multiplication(ExplicitOperation):
    """
    Multiplcation class. Subclass of M3Ls ExplicitOperation 
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='multiplication_operation')

    def compute(self):
        scalers = self.scalers
        arguments = self.arguments

        if f'x1' in scalers:
            x1 = scalers[f'x1']
            x2 = arguments[f'x2']
            csdl_model = csdl.Model()
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)
            y = x1 * x2_csdl

            csdl_model.register_output(name=self.output_name, var=y)

        elif f'x2' in scalers:
            x1 = arguments[f'x1']
            x2 = scalers[f'x2']
            csdl_model = csdl.Model()
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            y = x1_csdl * x2

            csdl_model.register_output(name=self.output_name, var=y)
        else:
            x1 = self.arguments[f'x1']
            x2 = self.arguments[f'x2']

            csdl_model = csdl.Model()
            x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
            x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)

            y = x1_csdl * x2_csdl
            csdl_model.register_output(name=self.output_name, var=y)

        return csdl_model
    
    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        random_name = generate_random_string()
        if isinstance(x1, (float, int, np.ndarray)):
            self.name = f'scaler_times_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.scalers[f'x1'] = x1
            self.arguments[f'x2'] = x2

            output = Variable(shape=x2.shape, operation=self)
            self.output_name = output.name
            

            # NOTE: in-line evaluations only work if all solver developers implement them
            # create csdl model for in-line evaluations
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x2'] = x2.value
            sim.run()
            output.value = sim[self.output_name]
            del sim
            # del operation_csdl
            # gc.collect()

        elif isinstance(x2, (float, int, np.ndarray)):
            self.name = f'{x1.name}_times_scaler_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.scalers[f'x2'] = x2
            self.arguments[f'x1'] = x1

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name
            
            # create csdl model for in-line evaluations
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x1'] = x1.value
            sim.run()
            output.value = sim[self.output_name]
            del sim
            # del operation_csdl
            # gc.collect()
        
        else:
            self.name = f'{x1.name}_times_{x2.name}_operation_{random_name}'
            self.arguments = {}
            self.scalers = {}
            self.arguments[f'x1'] = x1
            self.arguments[f'x2'] = x2

            output = Variable(shape=x1.shape, operation=self)
            self.output_name = output.name

            # create csdl model for in-line evaluations
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x1'] = x1.value
            sim[f'x2'] = x2.value
            sim.run()
            output.value = sim[self.output_name]

            del sim
            # del operation_csdl
            # gc.collect()

        return output

class Division(ExplicitOperation):
    """
    Division class. Subclass of M3Ls ExplicitOperation 
    """
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='division_operation')


    def compute(self):
        x1 = self.arguments[f'x1']
        x2 = self.arguments[f'x2']

        csdl_model = csdl.Model()
        
        # NOTE: can't divide by integer or float right now
        x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)

        if x1.shape != x2.shape:
            if (x1.shape != (1, ) or x1.shape != (1, 1)) and (x2.shape == (1, ) or x2.shape == (1, 1)):
                x2_csdl = csdl.expand(x2_csdl, shape=x1.shape)
            elif (x2.shape != (1, ) or x2.shape != (1, 1)) and (x1.shape == (1, ) or x1.shape == (1, 1)):
                x1_csdl = csdl.expand(x1_csdl, shape=x2.shape)
            else:
                raise ValueError(f"Cannot resolve shapes of division for variable shapes {x1.shape} and {x2.shape}")
        
        y = x1_csdl / x2_csdl
        
        csdl_model.register_output(name=self.output_name, var=y)
        return csdl_model
    
    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        random_name = generate_random_string()
        self.name = f'{x1.name}_division_{x2.name}_{random_name}_operation'
        # Define operation arguments
        self.arguments = {f'x1' : x1, f'x2' : x2}

        # Create the M3L variables that are being output
        output = Variable(shape=x1.shape, operation=self)
        self.output_name = output.name

        if (x1.value is not None) and (x2.value is not None):
            # create csdl model for in-line evaluations
            operation_csdl = self.compute()
            sim = Simulator(operation_csdl)
            sim[f'x1'] = x1.value
            sim[f'x2'] = x2.value
            sim.run()
            output.value = sim[self.output_name]

            del sim
            # del operation_csdl
            # gc.collect()

        return output
    

class Reshape(ExplicitOperation):
    '''
    Reshapes the variable to a new shape.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('name', types=str, default='reshape_operation')

    def assign_attributes(self):
        self.shape = self.parameters['shape']

    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        x = self.arguments['x']
        shape = self.shape

        operation_csdl = csdl.Model()
        x_csdl = operation_csdl.declare_variable(name='x', shape=x.shape)
        
        size = np.prod(x.value.shape)
        new_shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                size_others = np.prod(shape)/(-1)
                new_shape[i] = int(size/size_others)
                break
        shape = tuple(new_shape)
        self.shape = shape
        x_reshaped = csdl.reshape(x_csdl, shape)

        # self.output_name = replace_periods_with_underscores( f'{x.name}_reshaped')
        operation_csdl.register_output(name=self.output_name, var=x_reshaped)
        return operation_csdl

    def evaluate(self, x : Variable) -> Variable:
        '''
        User-facing method that the user will call to define a model evaluation.

        Parameters
        ----------
        mesh : Variable
            The mesh over which the function will be evaluated.

        Returns
        -------
        output : Variable
            The reshaped variable.
        '''
        random_string = generate_random_string()
        self.name = f'{x.name}_reshape_operation_{random_string}'
        if self.name == 'b_spline_hyper_volume_coefficients_reshape_operation':
            print(x.name)
            print(x.operation)
        self.parameters['name'] = self.name

        # Define operation arguments
        self.arguments = {'x' : x}

        output = Variable(shape=self.shape, operation=self)
        self.output_name = output.name

        operation_csdl = self.compute()
        output.shape = self.shape

        # Create the M3L variables that are being output
        # output_name = replace_periods_with_underscores(f'{x.name}_reshaped')
        # self.output_name = output.name

        
        # create csdl model for in-line evaluations
        sim = Simulator(operation_csdl)
        sim['x'] = x.value
        sim.run()
        output.value = sim[self.output_name]
        del sim
        # del operation_csdl
        # gc.collect()
        return output


class Sum(ExplicitOperation):
    """
    Sum class to perform the sum of an m3l variable.
    """
    def initialize(self, kwargs):
        self.parameters.declare('axes', types=tuple)

    def assign_attributes(self):
        self.axes = self.parameters['axes']

    def compute(self):
        x = self.arguments['x']

        csdl_model = csdl.Model()
        x_csdl = csdl_model.declare_variable(name='x', shape=x.shape)

        if len(x.shape) == len(self.axes):
            y = csdl.sum(x_csdl)
        else:
            y = csdl.sum(x_csdl, axes=self.axes)

        csdl_model.register_output(name=self.output_name, var=y)

        return csdl_model

    def evaluate(self, x : Variable) -> Variable:
        random_name = generate_random_string()
        self.name = f"sum_{x.name}_along_{self.axes[0]}_{random_name}_operation"
        self.arguments = {'x': x}

        output_shape = []
        for axis in range(len(x.shape)):
            if axis not in self.axes:
                output_shape.append(x.shape[axis])
        output_shape = tuple(output_shape)
        if not output_shape:
            output_shape = (1, )
        output = Variable(shape=output_shape, operation=self)
        self.output_name = output.name

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl, display_scripts=1)
        sim['x'] = x.value
        sim.run()
        output.value = sim[self.output_name]
        del sim
        # del operation_csdl
        # gc.collect()
        return output


class CrossProduct(ExplicitOperation):
    """
    Cross product class to perform the cross product of two m3l variables
    """
    def initialize(self, kwargs):
        self.parameters.declare('axis', types=int)

    def assign_attributes(self):
        self.axis = self.parameters['axis']

    def compute(self):
        # x1 = self.arguments[f'{self.output_name}_x1']
        # x2 = self.arguments[f'{self.output_name}_x2']
        x1 = self.arguments['x1']
        x2 = self.arguments['x2']


        csdl_model = csdl.Model()
        # x1_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x1', shape=x1.shape)
        # x2_csdl = csdl_model.declare_variable(name=f'{self.output_name}_x2', shape=x2.shape)

        x1_csdl = csdl_model.declare_variable(name=f'x1', shape=x1.shape)
        x2_csdl = csdl_model.declare_variable(name=f'x2', shape=x2.shape)

        y = csdl.cross(x1_csdl, x2_csdl, axis=self.axis)
        csdl_model.register_output(name=self.output_name, var=y)

        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable) -> Variable:
        random_name = generate_random_string()
        self.name = f"{x1.name}_cross_{x2.name}_{random_name}_operation"
        self.arguments = {'x1':x1, 'x2':x2}

        output = Variable(shape=x1.shape, operation=self)
        self.output_name = output.name
        # self.arguments = {f'{self.output_name}_x1': x1, f'{self.output_name}_x2' : x2}

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        # sim[f'{self.output_name}_x1'] = x1.value
        # sim[f'{self.output_name}_x2'] = x2.value
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        output.value = sim[self.output_name]

        del sim
        # del operation_csdl
        # gc.collect()
        
        return output


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
        # x1 = self.arguments['x1']
        # x2 = self.arguments['x2']
        # shape = x1.shape
        # shape[0] = x2.shape[0]
        operation_csdl = csdl.Model()
        csdl_vars = []
        for key, val in self.arguments.items():
            csdl_vars.append(operation_csdl.declare_variable(name=key, shape=val.shape))

        shape = self.shape
        # x1_csdl = operation_csdl.declare_variable(name='x1', shape=x1.shape)
        # x2_csdl = operation_csdl.declare_variable(name='x2', shape=x2.shape)
        y = operation_csdl.create_output(name=self.output_name, shape=shape)
        
        if len(shape) == 2:
            start_row_index = 0
            for i in range(len(csdl_vars)):
                csdl_var = csdl_vars[i]
                stop_row_index = csdl_var.shape[0]
                y[start_row_index:stop_row_index, :] = csdl_var
                start_row_index = stop_row_index
        elif len(shape) == 1:
            start_row_index = 0
            for i in range(len(csdl_vars)):
                csdl_var = csdl_vars[i]
                stop_row_index = csdl_var.shape[0] + start_row_index
                y[start_row_index:stop_row_index] = csdl_var
                start_row_index = stop_row_index

        # y[0:x1.shape[0],:] = x1_csdl
        # y[x1.shape[0]:,:] = x2_csdl
        # operation_csdl.register_output(name=self.output_name, var=y)
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

    def evaluate(self, x : Tuple[Variable]) -> Variable:
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
        if not isinstance(x, tuple):
            raise ValueError("Input must be a tuple")
        if len(x) <= 1:
            raise ValueError("Need at least two elements to perform Vstack operation")
        # if check_same_shape(t=x) is False:
        #     raise ValueError("All elements to be stacked must have the same shape")
        
        operation_name = 'v_stack_operation'
        self.arguments = {}
        for i, var in enumerate(x):
            operation_name += f'_{var.name}'
            self.arguments[f'x{i}'] = var

        self.name = operation_name

            

        # For 2-D arrays
        if len(x[0].shape) == 2:
            var_shape = x[0].shape
            # self.shape = (x1.shape[0] + x2.shape[0], ) + x1.shape[1:]
            self.shape = (int(var_shape[0] * (i+1)), ) + var_shape[1]
        # for 1-D vector
        elif len(x[0].shape) == 1:
            shape = 0 
            for var in x:
                shape += var.shape[0]
            self.shape = (shape, )
        else:
            print("Stacking of variables with shape > 2D not implemented")
            raise NotImplementedError
        
        # Create the M3L variables that are being output
        function_values = Variable(shape=self.shape, operation=self)
        self.output_name = function_values.name
        
         # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        for i, var in enumerate(x):
            sim[f'x{i}'] = var.value
        sim.run()
        function_values.value = sim[self.output_name]
        
        
        return function_values

def check_same_shape(t):
    first_array_shape = np.shape(t[0])

    for array in t[1:]:
        if np.shape(array) != first_array_shape:
            return False

    return True

class MatVec(ExplicitOperation):
    '''
    Class for the matvec product operation.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='dot_operation')
        self.parameters.declare('map', types=(np.ndarray, sps.csc_matrix, sps.coo_matrix))
        self.unique_name = ''

    def assign_attributes(self):
        self.map = self.parameters['map']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        map = self.map
        x = self.arguments['x']

        operation_csdl = csdl.Model()
        # map_csdl = operation_csdl.declare_variable(name='map', shape=map.shape, val=map.value.toarray())
        x_csdl = operation_csdl.declare_variable(name='x', shape=x.shape, val=x.value)

        b = csdl.matvec(map, x_csdl)

        operation_csdl.register_output(name=self.output_name, var=b)

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

    def evaluate(self, x:Variable) -> Variable:
        '''
        User-facing method that the user will call to define a model evaluation.

        Parameters
        ----------
        map : sps.csc_matrix
            The matrix in the matrix-vector product.
        x : Variable
            The vector in the matrix-vector product.

        Returns
        -------
        function_values : Variable
            The values of the function at the mesh locations.
        '''
        import m3l
        # if type(map) is np.ndarray or sps.isspmatrix(map):
        #     map_name = 'constant_map'
        #     map = m3l.Variable(name=map_name, shape=map.shape, operation=None, value=map)

        random_string = generate_random_string()
        # self.name = f'{map.name}_multiplied_with_{x.name}_operation'
        # self.name = f'{map.name}_multiplied_with_{x.name}_operation_{random_string}'
        self.name = f'{x.name}_matvec_operation_{random_string}'
        # Define operation arguments
        # self.arguments = {'map' : map, 'x' : x}
        self.arguments = {'x' : x}

        # Create the M3L variables that are being output
        output_shape = (self.map.shape[0],)
        output = Variable(shape=output_shape, operation=self)
        self.output_name = output.name
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        # import sys
        # print('matvec name size', sys.getsizeof(self.name)/1024/1024)
        # sim['map'] = map.value
        sim['x'] = x.value
        sim.run()
        output.value = sim[self.output_name]
        del sim
        # del operation_csdl
        # gc.collect()
        
        return output

class MatMat(ExplicitOperation):
    '''
    Class for the matrix-matrix product operation.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='dot_operation')
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        map = self.arguments['map']
        x = self.arguments['x']

        operation_csdl = csdl.Model()
        map_csdl = operation_csdl.declare_variable(name='map', shape=map.shape)
        x_csdl = operation_csdl.declare_variable(name='x', shape=x.shape)

        b = csdl.matmat(map_csdl, x_csdl)

        operation_csdl.register_output(name=self.output_name, var=b)

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

    def evaluate(self, map:Variable, x:Variable) -> Variable:
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
        import m3l
        if type(map) is np.ndarray or sps.isspmatrix(map):
            map_name = 'constant_map'
            map = m3l.Variable(name=map_name, shape=map.shape, operation=None, value=map)

        self.name = f'{map.name}_multiplied_with_{x.name}_operation'

        # Define operation arguments
        self.arguments = {'map' : map, 'x' : x}

        # Create the M3L variables that are being output
        output_shape = (map.shape[0],)
        output = Variable(shape=output_shape, operation=self)
        self.output_name = output.name
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['map'] = map.value
        sim['x'] = x.value
        sim.run()
        output.value = sim[self.output_name]
        del sim
        # del operation_csdl
        # gc.collect()

        return output


class Rotate(ExplicitOperation):
    '''
    Class for the rotate operation.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='rotate_operation')
        self.parameters.declare('units', types=str, default='degrees')

    def assign_attributes(self):
        self.units = self.parameters['units']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the rotation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        points = self.arguments['points']
        axis_origin = self.arguments['axis_origin']
        axis_vector = self.arguments['axis_vector']
        angles = self.arguments['angles']

        operation_csdl = csdl.Model()
        points_csdl = operation_csdl.declare_variable(name='points', shape=points.shape)
        axis_origin_csdl = operation_csdl.declare_variable(name='axis_origin', shape=axis_origin.shape)
        axis_vector_csdl = operation_csdl.declare_variable(name='axis_vector', shape=axis_vector.shape)
        angles_csdl = operation_csdl.declare_variable(name='angles', shape=angles.shape)

        num_points = np.prod(points.shape[:-1])
        num_angles = np.prod(angles.shape)

        normalized_axis = axis_vector_csdl / csdl.expand(csdl.pnorm(axis_vector_csdl), axis_vector_csdl.shape, 'i->ij')
        normalized_axis = csdl.reshape(normalized_axis, new_shape=(normalized_axis.shape[-1],))

        angles_flattened = csdl.reshape(angles_csdl, new_shape=((num_angles,))) # This rotate only works for 3D rotations

        # Translate control points into actuation origin frame
        axis_origin_csdl_expanded = csdl.expand(csdl.reshape(axis_origin_csdl, new_shape=(axis_origin_csdl.shape[-1],)),
                                                shape=(num_points,axis_origin_csdl.shape[-1]), indices='i->ji')
        points_origin_frame = points_csdl - axis_origin_csdl_expanded

        # Construct quaternion from rotation value
        rotated_points_flattened = operation_csdl.create_output(
                name='rotated_points_flattened', shape=(num_angles*num_points,) + points.shape[-1:])
        angle_counter = 0
        for i in range(len(angles.shape)):
            for t in range(angles.shape[i]):
                rotation_value = angles_flattened[angle_counter]
                if self.units == 'degrees':
                    rotation_value = rotation_value * np.pi/180

                quaternion = operation_csdl.create_output(f'quat_{t}', shape=(num_points,) + (4,))
                quaternion[:, 0] = csdl.expand(csdl.cos(rotation_value / 2), (num_points,) + (1,), 'i->ij')
                quaternion[:, 1] = csdl.expand(csdl.sin(rotation_value / 2) * normalized_axis[0], (num_points,) + (1,), 'i->ij')
                quaternion[:, 2] = csdl.expand(csdl.sin(rotation_value / 2) * normalized_axis[1], (num_points,) + (1,), 'i->ij')
                quaternion[:, 3] = csdl.expand(csdl.sin(rotation_value / 2) * normalized_axis[2], (num_points,) + (1,), 'i->ij')

                # Apply rotation
                rotated_points_origin_frame = csdl.quatrotvec(quaternion, points_origin_frame)

                # Translate rotated control points back into original coordinate frame
                rotated_points_t = rotated_points_origin_frame + axis_origin_csdl_expanded

                rotated_points_flattened[angle_counter*num_points:(angle_counter+1)*num_points,:] = rotated_points_t

                angle_counter += 1
        
        rotated_points = csdl.reshape(rotated_points_flattened, new_shape=(angles.shape + points.shape))

        operation_csdl.register_output(name=self.output_name, var=rotated_points)

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

    def evaluate(self, points:Variable, axis_origin:Variable, axis_vector:Variable, angles:Variable) -> Variable:
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
        import m3l

        if len(points.shape) == 1:
            print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
            points = points.reshape((-1,3))

        if type(points) is np.ndarray:
            points_name = 'constant_points'
            # points = m3l.Variable(name=points_name, shape=points.shape, operation=None, value=points)
            points = m3l.Variable(shape=points.shape, operation=None, value=points)

        if type(axis_origin) is np.ndarray:
            axis_origin_name = 'constant_axis_origin'
            # axis_origin = m3l.Variable(name=axis_origin_name, shape=axis_origin.shape, operation=None, value=axis_origin)
            axis_origin = m3l.Variable(shape=axis_origin.shape, operation=None, value=axis_origin)
        
        if type(axis_vector) is np.ndarray:
            axis_vector_name = 'constant_axis_vector'
            # axis_vector = m3l.Variable(name=axis_vector_name, shape=axis_vector.shape, operation=None, value=axis_vector)
            axis_vector = m3l.Variable(shape=axis_vector.shape, operation=None, value=axis_vector)

        if type(angles) is float or type(angles) is int:
            angles = m3l.Variable(shape=(1,), operation=None, value=angles)
        elif type(angles) is np.ndarray:
            angles_name = 'constant_angles'
            # angles = m3l.Variable(name=angles_name, shape=angles.shape, operation=None, value=angles)
            angles = m3l.Variable(shape=angles.shape, operation=None, value=angles)
        
        self.name = f'{points.name}_rotated_by_{angles.name}_about_{axis_vector.name}_at_point_{axis_origin.name}_operation'

        # Define operation arguments
        self.arguments = {'points' : points, 'axis_origin' : axis_origin, 'axis_vector':axis_vector, 'angles' : angles}

        # Create the M3L variables that are being output
        
        if len(angles.shape) > 1 or angles.shape[0] > 1:
            output_shape = angles.shape + points.shape
        else:
            output_shape = points.shape

        output = Variable(shape=output_shape, operation=self)
        self.output_name = output.name
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['points'] = points.value
        sim['axis_origin'] = axis_origin.value
        sim['axis_vector'] = axis_vector.value
        sim['angles'] = angles.value
        sim.run()
        output.value = sim[self.output_name]

        del sim
        # del operation_csdl
        # gc.collect()
        
        return output
    

class GetItem(ExplicitOperation):
    '''
    Class for the indexing operation.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='rotate_operation')
        self.parameters.declare('indices', types=tuple)

    def assign_attributes(self):
        self.indices = self.parameters['indices']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the indexing operation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        x = self.arguments['x']
        indices = self.indices

        operation_csdl = csdl.Model()
        x_csdl = operation_csdl.declare_variable(name='x', shape=x.shape)



        x_indexed = x_csdl[indices]


        operation_csdl.register_output(name=self.output_name, var=x_indexed)

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

    def evaluate(self, x:Variable) -> Variable:
        '''
        User-facing method that the user will call to index a Variable.

        Parameters
        ----------
        x : Variable
            The variable to be indexed.

        Returns
        -------
        function_values : Variable
            The values of the function at the mesh locations.
        '''

        self.name = f'{x.name}[{self.indices}]_operation'

        # Define operation arguments
        self.arguments = {'x' : x}

        # Create the M3L variables that are being output
        output_shape = tuple([len(self.indices)] + list(x.shape))

        output = Variable(shape=output_shape, operation=self)
        self.output_name = output.name
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x'] = x.value
        sim.run()
        output.value = sim[self.output_name]
        
        del sim
        # del operation_csdl
        # gc.collect()
        return output
    

if __name__ == "__main__":
    # Test Vstack
    var1 = Variable(shape=(4, ), value=1)
    var2 = Variable(shape=(21, ), value=2)
    var3 = Variable(shape=(9, ), value=3)
    var4 = Variable(shape=(7, ), value=4)
    var5 = Variable(shape=(13, ), value=5)
    var6 = Variable(shape=(2, ), value=6)
    var7 = Variable(shape=(10, ), value=7)

    vstack = VStack()

    v_stacked_vars = vstack.evaluate(x=(var1, var2, var3, var4, var5, var6, var7))

    print(v_stacked_vars)