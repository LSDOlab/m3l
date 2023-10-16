import csdl
from m3l.core.m3l_classes import ExplicitOperation, Variable
import numpy as np
import scipy.sparse as sps
from m3l.utils.utility_functions import replace_periods_with_underscores, generate_random_string
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
        
        if len(out_shape) == 0:
            out_shape = (1, )
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
        csdl_model.print_var(y)
        return csdl_model

    def evaluate(self, x1 : Variable, x2 : Variable):
        random_name = generate_random_string()
        self.name = f'{x1.name}_minus_{x2.name}_operation_{random_name}'
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
        
        # NOTE: can't divide by integer or float right now
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

        output_name = replace_periods_with_underscores( f'{x.name}_reshaped')
        operation_csdl.register_output(name=output_name, var=x_reshaped)
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
        self.name = f'{x.name}_reshaped_operation_to_{self.shape}'
        self.parameters['name'] = self.name

        # Define operation arguments
        self.arguments = {'x' : x}

        operation_csdl = self.compute()

        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{x.name}_reshaped')
        output = Variable(name=output_name, shape=self.shape, operation=self)
        
        # create csdl model for in-line evaluations
        sim = Simulator(operation_csdl)
        sim['x'] = x.value
        sim.run()
        output.value = sim[output_name]

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
        output = Variable(name=output_name, shape=x1.shape, operation=self)

        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x1'] = x1.value
        sim['x2'] = x2.value
        sim.run()
        output.value = sim[output_name]
        
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



class MatVec(ExplicitOperation):
    '''
    Class for the matvec product operation.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='dot_operation')
        self.unique_name = ''
    
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
        map_csdl = operation_csdl.declare_variable(name='map', shape=map.shape, val=map.value)
        x_csdl = operation_csdl.declare_variable(name='x', shape=x.shape, val=x.value)

        b = csdl.matvec(map_csdl, x_csdl*1)

        output_name = replace_periods_with_underscores(f'{map.name}_multiplied_with_{x.name}')
        operation_csdl.register_output(name=output_name, var=b)

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

        random_string = generate_random_string()
        # self.name = f'{map.name}_multiplied_with_{x.name}_operation'
        self.name = f'{map.name}_multiplied_with_{x.name}_operation_{random_string}'
        # Define operation arguments
        self.arguments = {'map' : map, 'x' : x}

        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(f'{map.name}_multiplied_with_{x.name}')
        output_shape = (map.shape[0],)
        output = Variable(name=output_name, shape=output_shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['map'] = map.value
        sim['x'] = x.value
        sim.run()
        output.value = sim[output_name]
        
        return output

class MatMat(ExplicitOperation):
    '''
    Class for the matvec product operation.
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

        output_name = replace_periods_with_underscores(f'{map.name}_multiplied_with_{x.name}')
        operation_csdl.register_output(name=output_name, var=b)

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
        output_name = replace_periods_with_underscores(f'{map.name}_multiplied_with_{x.name}')
        output_shape = (map.shape[0],)
        output = Variable(name=output_name, shape=output_shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['map'] = map.value
        sim['x'] = x.value
        sim.run()
        output.value = sim[output_name]
        
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

        output_name = replace_periods_with_underscores(
            f'{points.name}_rotated_by_{angles.name}_about_{axis_vector.name}_at_point_{axis_origin.name}')
        operation_csdl.register_output(name=output_name, var=rotated_points)

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

        if type(points) is np.ndarray:
            points_name = 'constant_points'
            points = m3l.Variable(name=points_name, shape=points.shape, operation=None, value=points)

        if type(axis_origin) is np.ndarray:
            axis_origin_name = 'constant_axis_origin'
            axis_origin = m3l.Variable(name=axis_origin_name, shape=axis_origin.shape, operation=None, value=axis_origin)
        
        if type(axis_vector) is np.ndarray:
            axis_vector_name = 'constant_axis_vector'
            axis_vector = m3l.Variable(name=axis_vector_name, shape=axis_vector.shape, operation=None, value=axis_vector)

        if type(angles) is float or type(angles) is int:
            angles = m3l.Variable(name='constant_angle', shape=(1,), operation=None, value=angles)
        elif type(angles) is np.ndarray:
            angles_name = 'constant_angles'
            angles = m3l.Variable(name=angles_name, shape=angles.shape, operation=None, value=angles)
        
        self.name = f'{points.name}_rotated_by_{angles.name}_about_{axis_vector.name}_at_point_{axis_origin.name}_operation'

        # Define operation arguments
        self.arguments = {'points' : points, 'axis_origin' : axis_origin, 'axis_vector':axis_vector, 'angles' : angles}

        # Create the M3L variables that are being output
        output_name = replace_periods_with_underscores(
            f'{points.name}_rotated_by_{angles.name}_about_{axis_vector.name}_at_point_{axis_origin.name}')
        
        if len(angles.shape) > 1 or angles.shape[0] > 1:
            output_shape = angles.shape + points.shape
        else:
            output_shape = points.shape

        output = Variable(name=output_name, shape=output_shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['points'] = points.value
        sim['axis_origin'] = axis_origin.value
        sim['axis_vector'] = axis_vector.value
        sim['angles'] = angles.value
        sim.run()
        output.value = sim[output_name]
        
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


        output_name = replace_periods_with_underscores(f'{x.name}[{self.indices}]')
        operation_csdl.register_output(name=output_name, var=x_indexed)

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
        output_name = replace_periods_with_underscores(f'{x.name}[{self.indices}]')
        
        output_shape = tuple([len(self.indices)] + list(x.shape))

        output = Variable(name=output_name, shape=output_shape, operation=self)
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        sim['x'] = x.value
        sim.run()
        output.value = sim[output_name]
        
        return output