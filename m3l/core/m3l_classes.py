from dataclasses import dataclass
from typing import Any, List

import numpy as np
import array_mapper as am
# import scipy.sparse as sps
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module


# @dataclass
# class Node:
#     '''
#     name : str
#         The name of the node.
#     '''
#     name : str

# @dataclass
# class Operation(Node):
#     '''
#     An M3L operation. This represents a mapping/model/operation/tranformation in the overall model.

#     Parameters
#     ----------
#     name : str
#         The name of the variable.
#     arguments : dict[str:Variable]
#         The dictionary of Variables that are arguments to the operation. The key is the string name of the variable in the operation.
#     '''
#     arguments : dict


class Operation(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.assign_attributes()  # Added this to make code more developer friendly (more familiar looking)
        
    def assign_attributes(self):
        pass


@dataclass
class CSDLOperation(Operation):
    '''
    An M3L CSDL operation. This represents a mapping/model/operation/tranformation in the overall model. The operation is represented
    as a CSDL model, making it a black box to M3L. This will be used to contain smaller/more basic operations.

    Parameters
    ----------
    name : str
        The name of the variable.
    arguments : dict[str:Variable]
        The dictionary of Variables that are arguments to the operation. The key is the string name of the variable in the operation.
    operation : csdl.Model
        The CSDL model that contains the operation.
    '''
    operation_csdl : csdl.Model


class ExplicitOperation(Operation):

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        pass
    
    def compute(self):
        '''
        Creates the CSDL model to compute the model output.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        pass

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

    def evaluate(self) -> tuple:
        '''
        User-facing method that the user will call to define a model evaluation.

        Parameters
        ----------
        TODO: This is solver-specific

        Returns
        -------
        model_outputs : tuple(List[m3l.Variable])
            The tuple of model outputs.
        '''
        pass
        # NOTE to solver developers: I recommend looking at an example such as aframe.


class ImplicitOperation(Operation):
    
    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        pass

    def evaluate_residuals(self):
        '''
        Solver developer API method for the backend to call.
        '''
        pass

    # only needed for dynamic stuff
    def compute_derivatives(self):
        '''
        Solver developer API method for the backend to call.
        '''
        pass

    # optional method
    def solve_residual_equations(self):
        '''
        Solver developer API method for the backend to call.
        '''
        pass

    def compute_invariant_matrix(self):
        '''
        Solver developer API method for the backend to call.
        '''
        pass

    def evaluate(self) -> tuple:
        '''
        User API method for the user/runscript to call.
        TODO: Replace this method header with information appropriate for user.
        '''
        pass



@dataclass
class Variable:
    '''
    An M3L variable. This represents information in the model.

    Parameters
    ----------
    name : str
        The name of the variable.
    shape : tuple
        The shape of the variable.
    operation : Opeation = None
        The operation that computes this variable. If none, this variable is a top-level input.
    value : np.ndarray = None
        The value of the variable.
    '''
    name : str
    shape : tuple
    operation : Operation = None
    value : np.ndarray = None

    def __add__(self, other):
        import m3l
        return m3l.add(self, other)


# @dataclass
# class NDArray:
#     '''
#     An n-dimensional array.

#     name : str
#         The name of the array.
#     shape : tuple
#         The shape of the array.
#     '''
#     name : str
#     shape : np.ndarray
#     operation : Operation = None
#     value : np.ndarray = None

class Add(ExplicitOperation):

    def initialize(self, kwargs):
        pass
    
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

        operation_csdl = csdl.Model()
        x1_csdl = operation_csdl.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = operation_csdl.declare_variable(name='x2', shape=x2.shape)
        y = x1_csdl + x2_csdl
        output_name = f'{x1.name}_plus_{x2.name}'
        operation_csdl.register_output(name=output_name, var=y)
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
        self.name = f'{x1.name}_plus_{x2.name}_operation'

        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}

        # Create the M3L variables that are being output
        function_values = Variable(name=f'{x1.name}_plus_{x2.name}', shape=x1.shape, operation=self)
        return function_values



@dataclass
class FunctionSpace:
    '''
    A class for representing function spaces.
    '''
    # reference_geometry : Function = None
    pass    # do we want separate class for state functions that point to a reference geometry?


@dataclass
class Function:
    '''
    A class for representing a general function.

    Parameters
    ----------
    name : str
        The name of the function.
    function_space : FunctionSpace
        The function space from which this function is defined.
    coefficients : NDarray = None
        The coefficients of the function.
    '''
    name : str
    space : FunctionSpace
    coefficients : Variable = None

    def __call__(self, mesh : am.MappedArray) -> Variable:
        return self.evaluate(mesh)

    def evaluate(self, mesh : am.MappedArray) -> Variable:
        '''
        Evaluate the function at a given set of nodal locations.

        Parameters
        ----------
        mesh : am.MappedArray
            The mesh to evaluate over.

        Returns
        -------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        # num_values = np.prod(mesh.shape[:-1])
        # num_coefficients = np.prod(self.coefficients.shape[:-1])
        # temp_map = np.eye(num_values, self.function_space.num_coefficients)
        # temp_map[self.function_space.num_coefficients:,0] = np.ones((num_values-self.function_space.num_coefficients,))
        # output_name = f'nodal_{self.name}'
        # output_shape = tuple(mesh.shape[:-1]) + (self.coefficients.shape[-1],)

        # # csdl_map = csdl.Model()
        # csdl_map = ModuleCSDL()
        # function_coefficients = csdl_map.register_module_input(self.coefficients.name, shape=(num_coefficients, self.coefficients.shape[-1]))
        # map_csdl = csdl_map.create_input(f'{self.name}_evaluation_map', temp_map)
        # flattened_function_values_csdl = csdl.matmat(map_csdl, function_coefficients)
        # function_values_csdl = csdl.reshape(flattened_function_values_csdl, new_shape=output_shape)
        # csdl_map.register_output(output_name, function_values_csdl)

        # # parametric_coordinates = self.function_space.reference_geometry.project(mesh.value, return_parametric_coordinates=True)
        # # map = self.function_space.compute_evaluation_map(parametric_coordinates)

        # # Idea: Have the MappedArray store the parametric coordinates so we don't have to repeatendly perform projection.

        # # function_values = Variable(name=output_name, upstream_variables={self.name:self}, map=csdl_map, mesh=mesh)

        # evaluate_operation = CSDLOperation(name=f'{self.name}_evaluation', arguments={self.coefficients.name: self.coefficients}, operation_csdl=csdl_map)
        # function_values = Variable(name=output_name, shape=output_shape, operation=evaluate_operation)
        # return function_values

        function_evaluation_model = FunctionEvaluation(function=self, mesh=mesh)
        function_values = function_evaluation_model.evaluate(self.coefficients)
        return function_values
    
    def inverse_evaluate(self, function_values:Variable):
        '''
        Performs an inverse evaluation to set the coefficients of this function given an input of evaluated points over a mesh.

        Parameters
        ----------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        # map = Perform B-spline fit and potentially some sort of conversion from extrinsic to intrinsic

        # num_values = np.prod(function_values.mesh.shape[:-1])
        num_values = np.prod(function_values.shape[:-1])
        temp_map = np.eye(self.function_space.num_coefficients, num_values)

        # csdl_map = csdl.Model()
        csdl_map = ModuleCSDL()
        function_values_csdl = csdl_map.register_module_input(function_values.name, shape=(num_values,3))
        map_csdl = csdl_map.create_input(f'{self.name}_inverse_evaluation_map', temp_map)
        function_coefficients_csdl = csdl.matmat(map_csdl, function_values_csdl)
        csdl_map.register_output(f'{self.name}_coefficients', function_coefficients_csdl)

        coefficients_shape = (temp_map.shape[0],3)
        operation = CSDLOperation(name=f'{self.name}_inverse_evaluation', arguments=[function_values], operation_csdl=csdl_map)
        function_coefficients = Variable(name=f'{self.name}_coefficients', shape=coefficients_shape, operation=operation)
        return function_coefficients


class FunctionEvaluation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('function', types=Function)
        self.parameters.declare('mesh', types=am.MappedArray)

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.function = self.parameters['function']
        self.mesh = self.parameters['mesh']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''

        mesh = self.mesh
        function_space = self.function.space
        coefficients = self.arguments['coefficients']

        num_values = np.prod(mesh.shape[:-1])
        num_coefficients = np.prod(coefficients.shape[:-1])
        temp_map = np.eye(num_values, function_space.num_coefficients)
        temp_map[function_space.num_coefficients:,0] = np.ones((num_values-function_space.num_coefficients,))
        output_name = f'evaluated_{self.function.name}'
        output_shape = tuple(mesh.shape[:-1]) + (coefficients.shape[-1],)

        csdl_map = ModuleCSDL()
        function_coefficients = csdl_map.register_module_input(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
                                                                val=coefficients.value.reshape((-1, coefficients.shape[-1])))
        map_csdl = csdl_map.create_input(f'{self.name}_evaluation_map', temp_map)
        flattened_function_values_csdl = csdl.matmat(map_csdl, function_coefficients)
        function_values_csdl = csdl.reshape(flattened_function_values_csdl, new_shape=output_shape)
        csdl_map.register_output(output_name, function_values_csdl)
        return csdl_map

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

    def evaluate(self, coefficients:Variable) -> tuple:
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
        self.name = f'{self.function.name}_evaluation'

        # Define operation arguments
        self.arguments = {'coefficients' : coefficients}

        # Create the M3L variables that are being output
        function_values = Variable(name=f'evaluated_{self.function.name}', shape=self.mesh.shape, operation=self)
        return function_values



class Model:   # Implicit (or not implicit?) model groups should be an instance of this
    '''
    A class for storing a group of M3L models. These can be used to establish coupling loops.
    '''

    def __init__(self) -> None:
        '''
        Constructs a model group.
        '''
        self.models = {}
        self.operations = {}
        self.outputs = {}
        self.parameters = None

    # def add(self, submodel:Model, name:str):
    #     self.models[name] = submodel

    def register_output(self, output:Variable):
        '''
        Registers a state to the model group so the model group will compute and output this variable.
        If inverse_evaluate is called on a variable that already has a value, the residual is identified
        and an implicit solver is used.

        Parameters
        ----------
        output : Variable
            The variable that the model will output.
        '''
        if isinstance(output, dict):
            for key, value in output.items():
                self.outputs[value.name] = value
        elif type(output) is Variable:
            self.outputs[output.name] = output
        else:
            print(type(output))
            raise NotImplementedError
        # self.outputs[output.name] = output


    def set_linear_solver(self, linear_solver:csdl.Solver):
        '''
        Sets a linear solver for the model group to resolve couplings.

        Parameters
        ----------
        solver : csdl.Solver
            The solver object.
        '''
        self.linear_solver = linear_solver

    def set_nonlinear_solver(self, nonlinear_solver:csdl.Solver):
        '''
        Sets a nonlinear solver for the model group to resolve couplings.

        Parameters
        ----------
        solver : csdl.Solver
            The solver object.
        '''
        self.nonlinear_solver_solver = nonlinear_solver


    def gather_operations(self, variable:Variable):
        if variable.operation is not None:
            operation = variable.operation
            for input_name, input in operation.arguments.items():
                self.gather_operations(input)

            if operation.name not in self.operations:
                self.operations[operation.name] = operation


    # def assemble(self):
    #     # Assemble output states
    #     for output_name, output in self.outputs.items():
    #         self.gather_operations(output)
        
    #     model_csdl = ModuleCSDL()

    #     for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process

    #         if type(operation.operation_csdl) is csdl.Model:
    #             model_csdl.add(submodel=operation.operation_csdl, name=operation.name, promotes=[]) # should I suppress promotions here?
    #         else: # type(operation.operation_csdl) is ModuleCSDL:
    #             model_csdl.add_module(submodule=operation.operation_csdl, name=operation.name, promotes=[]) # should I suppress promotions here?
            

    #         for arg_name, arg in operation.arguments.items():
    #             if arg.operation is not None:
    #                 model_csdl.connect(arg.operation.name+"."+arg.name, operation_name+"."+arg_name)  # Something like this for no promotions
    #                 # if arg.name == arg_name:
    #                 #     continue    # promotion will automatically connect if the names match
    #                 # else:
    #                 #     model_csdl.connect(arg.name, arg_name)  # If names don't match, connect manually

    #     self.csdl_model = model_csdl
    #     return self.csdl_model
    

    def assemble(self):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        model_csdl = ModuleCSDL()

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process

            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute()

                if type(operation_csdl) is csdl.Model:
                    model_csdl.add(submodel=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
                elif issubclass(type(operation_csdl), ModuleCSDL):
                    model_csdl.add_module(submodule=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None:
                        model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting


        self.csdl_model = model_csdl
        return self.csdl_model


    def assemble_csdl(self) -> ModuleCSDL:
        self.assemble()

        return self.csdl_model
