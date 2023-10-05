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

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the residuals.
        '''
        pass

    # only needed for dynamic stuff
    def compute_derivatives(self):
        '''
        Solver developer API method for the backend to call.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the derivatives of the model/operation outputs.
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

        Returns
        -------
        invariant_matrix_csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the invariant matrix for the SIFR methodology.
        '''
        pass

    def evaluate(self) -> tuple:
        '''
        User API method for the user/runscript to call.
        TODO: Replace this method header with information appropriate for user.

        Parameters
        ----------
        TODO: This is solver-specific

        Returns
        -------
        model_outputs : tuple(List[m3l.Variable])
            The tuple of model outputs.
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
    coefficients : Variable

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

    def __init__(self, name:str=None, arguments:dict[str,Variable]=None, outputs:dict[str,Variable]=None,
                 submodels:dict=None, operations:list[Variable]=None) -> None:
        '''
        Constructs a model.
        '''
        self.name = name
        self.arguments = arguments
        self.outputs = outputs
        self.submodels = submodels
        self.operations = operations
        
        if self.arguments is None:
            self.arguments = {}
        if self.outputs is None:
            self.outputs = {}
        if self.submodels is None:
            self.submodels = {}
        if self.operations is None:
            self.operations = []
        
        self.csdl_model = None

    # def add(self, submodel, name:str):
    #     self.models[name] = submodel

    # NOTE: THIS SHOULDN'T BE NEEDED BECAUSE THE GRAPH IS IN THE RUN SCRIPT
    # def connect(self, connect_from:Variable, connect_to:Variable):
    #     pass

    def define_submodel(self, name:str, input_variables:list[Variable], output_variables:list[Variable]):
        '''
        Defines a submodel to include the operations that compute the output variables from the input variables.

        name : str
            The name of the submodel
        input_variables: list[Variable]
            The input variables to this submodel
        output_variables: list[Variable]
            The output variables to this submodel
        '''
        arguments = {}
        for input in input_variables:
            arguments[input.name] = input

        outputs = {}
        for output in output_variables:
            outputs[output.operation.name + '.' + output.name] = output

        submodel = Model(name=name, arguments=arguments, outputs=outputs)
        self.submodels[name] = submodel
        # NOTE: There needs to be some sort of .copy() on this returned submodel so the operations don't point to the same place in memory
        return submodel     # IDEA: The user can use this to define a submodel, then "copy/paste" it to use it multiple times in the run script.

    def register_output(self, output:Variable, name:str=None):
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
                self.outputs[value.operation.name + '.' + value.name] = value
        elif type(output) is Variable:
            self.outputs[output.operation.name + '.' + output.name] = output
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


    def check_if_variable_is_submodel_output(self, variable:Variable):
        '''
        Checks if an variable is the output of a stored submodel of this model.
        '''
        for submodel_name, submodel in self.submodels.items():
            for submodel_output_name, submodel_output in submodel.outputs.items():
                if variable is submodel_output:
                    return submodel
        return False
    
    def check_if_variable_is_argument(self, variable:Variable):
        '''
        Checks if an variable is an argument to this model.
        '''
        for argument_name, argument in self.arguments.items():
            if variable is argument:
                return True
        return False
    
    def check_if_operation_is_in_model(self, operation:Operation):
        '''
        Checks if an explicit operation is within the scope of this model.
        '''
        for model_operation in self.operations:
            if issubclass(type(model_operation), Model):
                if model_operation.check_if_operation_is_in_model(operation):
                    return True
            else:
                if operation is model_operation:
                    return True
                
    def check_if_operation_has_been_added(self, operation:Operation):
        '''
        Checks if this operation has already been added to the model.
        '''
        for model_operation in self.operations:
            if operation is model_operation:
                return True
            
    def get_operation_that_variable_is_input_to(self, variable:Variable):
        '''
        Gets the operations that computes a variable
        '''
        for operation in self.operations:
            for argument_name, argument in operation.arguments.items():
                if variable is argument:
                    if issubclass(type(operation), Model):
                        operation.get_operation_that_variable_is_input_to(variable)
                    else:
                        return operation, argument_name


    def gather_operations(self, variable:Variable):
        is_argument = self.check_if_variable_is_argument(variable)
        if not is_argument and (variable.operation is not None):
            submodel = self.check_if_variable_is_submodel_output(variable)
            if submodel:
                # for submodel_output_name, submodel_output in submodel.outputs.items():
                #     submodel.gather_operations(submodel_output)
                submodel.assemble()

                operation = submodel
                # self.operations[submodel.name] = submodel
            else:
                operation = variable.operation
            # operation = variable.operation

            # for input_name, input in operation.arguments.items():
            for input_name, input in operation.arguments.items():
                self.gather_operations(input)

            operation_is_added = self.check_if_operation_has_been_added(operation)
            # if operation.name not in self.operations:
            #     self.operations[operation.name] = operation
            if not operation_is_added:
                self.operations.append(operation)
    

    def assemble(self):
        if self.csdl_model is not None:
            return  # Don't waste time reassembling models that are already assembled

        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        model_csdl = ModuleCSDL()

        for operation in self.operations:   # Already in correct order due to recursion process

            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute()

                if type(operation_csdl) is csdl.Model:
                    model_csdl.add(submodel=operation_csdl, name=operation.name, promotes=[])
                elif issubclass(type(operation_csdl), ModuleCSDL):
                    model_csdl.add_module(submodule=operation_csdl, name=operation.name, promotes=[])
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    operation_is_in_model = self.check_if_operation_is_in_model(input.operation)
                    if input.operation is not None and operation_is_in_model:
                        submodel = self.check_if_variable_is_submodel_output(input)
                        connect_from = input.operation.name+'.'+input.name
                        connect_to = operation.name+"."+input_name
                        if submodel:
                            connect_from = submodel.name+'.'+connect_from   # NOTE: THIS ONLY WORKS FOR ONE LEVEL OF SUBMODELS
                        # NOTE: ALSO, IF WE HAVE MULTIPLE LEVELS ON THE MODEL NESTING, WE ONLY WANT ONE CONNECTION, NOT ONE PER LEVEL
                        #   One idea is to only do the inner-model connections here (operation -> operation) and do all of the model connections
                        #       from the top level. Follow-up: I am kind of doing this, but the naming still only works one level.
                        model_csdl.connect(connect_from, connect_to)
            elif issubclass(type(operation), ImplicitOperation):
                # TODO
                pass

            # General strategy:
            #   Create a submodel heirarchy according to user definitions of inputs/outputs. The boundaries of these submodels cannot overlap.
            #   When creating the m3l model's csdl model, nest the appropriate operations in csdl model's to match the heirarchy. This will
            #   make sure the namespace is unique (both csdl and m3l). On m3l side, need to append to dictionary name.
            #   IDEA: When assembling at the model level, check if the string has a . in it to see if it's in a model. In gathering time,
            #       I need to append that to the dictionary name.

            # NOTE: I don't think the models should show up here as operations. ACTUALLY, NOW I THINK THEY SHOULD?
            elif issubclass(type(operation), Model):
                # First establish explicit model heirarchy, then figure out implicit model heirarchy.
                # Do we want recursion into the model heirarchy?
                # Objects should not store their parents. Only parents storing children (this makes more sense from an object definition
                # and object orientated programming philosophy). In MappedArrays, subsequent arrays have nothing to do with the definition of
                # an array in question (which would be the parent to subsequent arrays), while a Model is definitely defined by its children.

                # COMPLETE 180!! The thing with M3L is that we are taking a variable-centric approach, which is the opposite from CSDL which 
                # is a model-centric approach.
                operation_csdl = operation.csdl_model
                model_csdl.add_module(submodule=operation_csdl, name=operation.name, promotes=[])

                for _, input in operation.arguments.items():
                    # continue
                    operation_is_in_model = self.check_if_operation_is_in_model(input.operation)
                    if input.operation is not None and operation_is_in_model:
                        connect_from = input.operation.name+"."+input.name
                        operation_that_this_variable_feeds_to_and_name = operation.get_operation_that_variable_is_input_to(input)
                        if operation_that_this_variable_feeds_to_and_name is not None:
                            operation_that_this_variable_feeds_to = operation_that_this_variable_feeds_to_and_name[0]
                            input_name = operation_that_this_variable_feeds_to_and_name[1]
                        else:
                            continue    # Not sure if this is correct, but currently ignoring arguments that aren't used.
                        model_name = operation.name
                        operation_name = operation_that_this_variable_feeds_to.name
                        connect_to = model_name+'.'+operation_name+'.'+ input_name

                        submodel = self.check_if_variable_is_submodel_output(input)
                        if submodel:
                            connect_from = submodel.name+'.'+connect_from   # NOTE: THIS ONLY WORKS FOR ONE LEVEL OF SUBMODELS
                        # NOTE: ALSO, IF WE HAVE MULTIPLE LEVELS ON THE MODEL NESTING, WE ONLY WANT ONE CONNECTION, NOT ONE PER LEVEL
                        model_csdl.connect(connect_from, connect_to)

            # elif issubclass(type(operation), ImplicitModel):
            #     # Want this to add this csdl model, resolve coupling, and make connections.
            #     # I think the gather_operations needs to also store the heirarchy so we know how to make connections.
            #     operation_csdl = operation.assemble_csdl()
            #     model_csdl.add(submodel=operation_csdl, name=operation.name, promotes=[])

        self.csdl_model = model_csdl
        return self.csdl_model


    def assemble_csdl(self) -> ModuleCSDL:
        self.assemble()

        return self.csdl_model
    
    def evaluate(self, inputs:list[Variable]):
        '''
        This is meant for submodels. This takes in the list of arguments and outputs the outputs.

        TODO: The outputs need to be copied/remade to have this model as the operation that computes it.
        '''
        return self.outputs
    

@dataclass
class Submodel:
    name : str
    model : Model
    inputs : list[Variable]
    outputs : list[Variable]
    operations : list[Operation] = None
    submodels : list = None # list of submodels # NOTE: Start with no nesting of submodels.

    def gather_my_operations(self):
        for output_name, output in self.outputs.items():
            self.gather_operations(output)

    def gather_operations(self, variable:Variable):
        if variable.operation is not None:
            operation = variable.operation
            for input_name, input in operation.arguments.items():
                self.gather_operations(input)

            if operation not in self.operations:    # need to properly check
                self.operations[operation.name] = operation

        # for operation_input_name, operation_input in variable.arguments.items():
        #     self.operations[operation_input_name] = operation_input
        #     # use recursion
        
        #     if operation_input is submodel_input:
        #         continue (exit branch of computational tree)


class ImplicitModel:
    
    def __init__(self) -> None:
        pass


    def assemble_csdl(self):
        pass

