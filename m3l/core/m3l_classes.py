from dataclasses import dataclass
from typing import Any, List

import numpy as np
import array_mapper as am
# import scipy.sparse as sps
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module

''' Classes representing spaces '''
@dataclass
class Field:
    '''
    A class for representing mathematical fields such as the field of all real numbers.
    '''
    pass    # This may potentially have more information in the future

@dataclass
class VectorSpace:
    '''
    A class for representing vector spaces.

    Parameters
    ----------
    field : Field
        The field that the vector space is defined over.
    dimensions : tuple
        The dimensions/shape of the vectors.
    '''
    field : Field
    dimensions : tuple

@dataclass
class FunctionSpace:
    '''
    A class for representing function spaces.
    '''
    # reference_geometry : Function = None
    pass    # do we want separate class for state functions that point to a reference geometry?


''' Classes functions and information '''
@dataclass
class Variable:
    '''
    A general M3L variable.

    Parameters
    ----------
    name : str
        The name of the variable.
    upstream_variables : dict = None
        A dictionary containing the immediate inputs to the map to compute this variable.
    map : csdl.Model = None
        A CSDL Model that computes this variable given input of the upstream variables.
    value : np.ndarray = None
        The value of the variable.
    '''
    name : str
    upstream_variables : dict = None
    map : csdl.Model = None
    value : np.ndarray = None

@dataclass
class Vector:
    '''
    A general vector class.

    Parameters
    ----------
    name : str
        The name of the variable.
    upstream_variables : dict = None
        A dictionary containing the immediate inputs to the map to compute this variable.
    map : csdl.Model = None
        A CSDL Model that computes this variable given input of the upstream variables.
    value : np.ndarray = None
        The value of the variable.
    vector_space : VectorSpace
        The vector space from which this vector is defined.
    '''
    name : str
    vector_space : VectorSpace
    upstream_variables : dict = None
    map : csdl.Model = None
    value : np.ndarray = None

    def __post_init__(self):
        self.value = self.values

@dataclass
class FunctionValues:
    '''
    A class for representing the evaluation of a function over a mesh.

    Parameters
    ----------
    name : str
        The name of the variable.
    upstream_variables : dict = None
        A dictionary containing the immediate inputs to the map to compute this variable.
    map : csdl.Model = None
        A CSDL Model that computes this variable given input of the upstream variables.
    value : np.ndarray = None
        The value of the variable.
    mesh : am.MappedArray
        The mesh over which the function is evaluated.
    '''
    name : str
    mesh : am.MappedArray
    upstream_variables : dict = None
    map : csdl.Model = None
    value : np.ndarray = None

    def __post_init__(self):
        self.values = self.value

@dataclass
class Function:
    '''
    A class for representing a general function.

    Parameters
    ----------
    name : str
        The name of the variable.
    upstream_variables : dict = None
        A dictionary containing the immediate inputs to the map to compute this variable.
    map : csdl.Model = None
        A CSDL Model that computes this variable given input of the upstream variables.
    function_space : FunctionSpace
        The function space from which this function is defined.
    coefficients : np.ndarray = None
        The coefficients of the function.
    '''
    name : str
    function_space : FunctionSpace
    upstream_variables : dict = None
    map : csdl.Model = None
    coefficients : np.ndarray = None

    def __call__(self, mesh : am.MappedArray) -> FunctionValues:
        return self.evaluate(mesh)

    def evaluate(self, mesh : am.MappedArray) -> FunctionValues:
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
        num_values = np.prod(mesh.shape[:-1])
        temp_map = np.eye(num_values, self.function_space.num_coefficients)
        temp_map[self.function_space.num_coefficients:,0] = np.ones((num_values-self.function_space.num_coefficients,))
        output_name = f'nodal_{self.name}'

        # csdl_map = csdl.Model()
        csdl_map = ModuleCSDL()
        function_coefficients = csdl_map.declare_variable(self.name, shape=(self.function_space.num_coefficients,3))
        map_csdl = csdl_map.create_input(f'{self.name}_evaluation_map_over', temp_map)
        function_values_csdl = csdl.matmat(map_csdl, function_coefficients)
        csdl_map.register_output(output_name, function_values_csdl)

        # parametric_coordinates = self.function_space.reference_geometry.project(mesh.value, return_parametric_coordinates=True)
        # map = self.function_space.compute_evaluation_map(parametric_coordinates)

        # Idea: Have the MappedArray store the parametric coordinates so we don't have to repeatendly perform projection.

        function_values = FunctionValues(name=output_name, upstream_variables={self.name:self}, map=csdl_map, mesh=mesh)
        return function_values
    
    def inverse_evaluate(self, function_values:FunctionValues):
        '''
        Performs an inverse evaluation to set the coefficients of this function given an input of evaluated points over a mesh.

        Parameters
        ----------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        # map = Perform B-spline fit and potentially some sort of conversion from extrinsic to intrinsic

        num_values = np.prod(function_values.mesh.shape[:-1])
        temp_map = np.eye(self.function_space.num_coefficients, num_values)

        # csdl_map = csdl.Model()
        csdl_map = ModuleCSDL()
        function_values_csdl = csdl_map.declare_variable(function_values.name, shape=(num_values,3))
        map_csdl = csdl_map.create_input(f'{self.name}_inverse_evaluation_map', temp_map)
        function_coefficients_csdl = csdl.matmat(map_csdl, function_values_csdl)
        csdl_map.register_output(f'{self.name}_coefficients', function_coefficients_csdl)

        self.upstream_variables = {function_values.name:function_values}
        self.map = csdl_map
        return self


''' Classes for representing models '''
@dataclass
class ModelIOModule:
    '''
    A parent class for representing modules that map variables in or out of a model.

    Parameters
    ----------
    name : str
        The name of the mapping.
    map : {np.ndarray, sps.csc_matrix}
        The map that maps the variable in or out of the solver.
    '''
    name : str
    map : np.ndarray

@dataclass
class ModelInputModule(ModelIOModule):
    '''
    A class for representing modules that map a variable into a model.

    Parameters
    ----------
    name : str
        The name of the mapping.
    map : {np.ndarray, sps.csc_matrix}
        The map that maps the variable in or out of the solver.
    module_input : FunctionValues
        The input that will be mapped into the model.
    model_input_name : str
        The name of input as the model's csdl model is expecting.
    '''
    module_input : FunctionValues
    model_input_name : str

@dataclass
class ModelOutputModule(ModelIOModule):
    '''
    A class for representing modules that map a variable out of a model.

    Parameters
    ----------
    name : str
        The name of the mapping.
    map : {np.ndarray, sps.csc_matrix}
        The map that maps the variable in or out of the solver.
    model_output_name : str
        The name of the output as the model's csdl model is outputing.
    module_output_name : str
        The name that will be given to the evaluated function values.
    module_output_mesh : am.MappedArray
        The mesh that the function values are evaluated over.
    '''
    model_output_name : str
    module_output_name : str
    module_output_mesh : am.MappedArray

class Model(Module):    # Solvers should be an instance of this
    '''
    Parent model class that solvers/models should inherit from.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.assign_attributes()  # Added this to make code more developer friendly (more familiar looking)
        
    def assign_attributes(self):
        pass

    # def _assemble_csdl(self, input:Function, output_mesh:am.MappedArray=None):
    #     model_csdl = self.model._assemble_csdl()
    #     self.model.construct_map_in(input)
    #     self.model.construct_map_out(output_mesh)

    #     model_name_camel = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', type(model_csdl).__name__)
    #     model_name_snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', model_name_camel).lower()
    #     map_in_csdl = self.model.map_in_csdl
    #     # if map_in_csdl is None:
    #     #     pass
    #     #     map_in_csdl = identity
    #     map_out_csdl = self.model.map_out_csdl
    #     # if map_in_csdl is None:
    #     #     map_out_csdl = identity

    #     csdl_model = ModuleCSDL()
    #     csdl_model.add(map_in_csdl, 'map_in')
    #     csdl_model.add_module(model_csdl, model_name_snake)
    #     csdl_model.add(map_out_csdl, 'map_out')
    #     self.csdl_model = csdl_model

    #     return csdl_model
    
    def construct_module_csdl(self, model_map:csdl.Model, input_modules:List[ModelInputModule], output_modules:List[ModelOutputModule]):
        '''
        Automates a significant portion of a model's evaluate method.

        Parameters
        ----------
        model_map : csdl.Model
            The csdl model for the solver/model.
        input_modules : List[ModelInputModule]
            The list of input modules that map into the solver/model.
        output_modules : List[ModelOutputModule]
            The list of output modules that map out of the solver/model.

        Returns
        -------
        outputs : tuple
            A tuple containing the FunctionValues object corresponding to each output module.
        '''
        
        module_csdl = ModuleCSDL()

        inputs_dictionary = {}
        input_mappings_csdl = csdl.Model()
        for input_module in input_modules:
            module_input = input_module.module_input
            map = input_module.map
            model_input_name = input_module.model_input_name
            if module_input is None:
                continue

            if module_input is not None:
                num_variable = np.prod(module_input.mesh.shape[:-1])
                module_input_csdl = input_mappings_csdl.declare_variable(name=module_input.name, shape=(num_variable, 3)) # 3 hardcoded because need info
                map_csdl = input_mappings_csdl.create_input(f'{input_module.name}_map', val=map)
                model_input_csdl = csdl.matmat(map_csdl, module_input_csdl)
                input_mappings_csdl.register_output(model_input_name, model_input_csdl)

                inputs_dictionary[module_input.name] = module_input

        output_mappings_csdl = csdl.Model()
        for output_module in output_modules:
            model_output_name = output_module.model_output_name
            map = output_module.map
            module_output_name = output_module.module_output_name

            model_output_csdl = output_mappings_csdl.declare_variable(name=model_output_name, shape=(map.shape[-1],3))  # 3 hardcoded because need info
            map_csdl = output_mappings_csdl.create_input(f'{output_module.name}_map', val=map)
            module_output_csdl = csdl.matmat(map_csdl, model_output_csdl)
            output_mappings_csdl.register_output(module_output_name, module_output_csdl)


        module_csdl.add(submodel=input_mappings_csdl, name='inputs_module')
        module_csdl.add(submodel=model_map, name='model')
        module_csdl.add(submodel=output_mappings_csdl, name='outputs_module')

        outputs = []
        for output_module in output_modules:
            output = FunctionValues(name=output_module.module_output_name,
                                    upstream_variables=inputs_dictionary,
                                    map=module_csdl,
                                    mesh=output_module.module_output_mesh)
            outputs.append(output)
        
        return tuple(outputs)


class ModelGroup:   # Implicit (or not implicit?) model groups should be an instance of this
    '''
    A class for storing a group of M3L models. These can be used to establish coupling loops.
    '''

    def __init__(self) -> None:
        '''
        Constructs a model group.
        '''
        self.models = {}
        self.variables = {}
        self.outputs = {}
        self.parameters = None

    # def add(self, submodel:Model, name:str):
    #     self.models[name] = submodel

    def register_output(self, output:Function):
        '''
        Registers a state to the model group so the model group will compute and output this variable.
        If inverse_evaluate is called on a variable that already has a value, the residual is identified
        and an implicit solver is used.

        Parameters
        ----------
        output : Function
            The function for which the model group will output its coefficients.
        '''
        self.outputs[output.name] = output


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

    # def assemble_state(self, state):
    #     if state.absolute_map is not None:
    #         return  # already assembled
        
    #     if state.upstream_states is None:
    #         return  # This is an input and doesn't need to be assembled. It also doesn't have a mapping associated with it.

    #     # Recursively assemble upstream states first
    #     for upstream_state in state.upstream_states:
    #         self.assemble_state(upstream_state)

    #     absolute_map = ModuleCSDL()
    #     for upstream_state in state.upstream_states:
    #         if upstream_state.upstream_states is not None:
    #             absolute_map.add_module(submodule=upstream_state.absolute_map, name=f'{upstream_state.name}_model')
    #         else:
    #             # upstream state is an absolute input
    #             num_upstream_state_values = np.cumprod(upstream_state.shape[:-1])[-1]
    #             upstream_state_flattened_shape = (num_upstream_state_values, upstream_state.shape[-1])
    #             absolute_map.register_module_input(upstream_state.name, shape=upstream_state_flattened_shape)

    #     if state.relative_map is not None:
    #         absolute_map.add_module(submodule=state.relative_map, name=f'{state.name}_relative_map')

    #     state.absolute_map = absolute_map

    def gather_variables(self, variable:Function):
        if variable.upstream_variables is not None:
            for upstream_variable_name, upstream_variable in variable.upstream_variables.items():
                self.gather_variables(upstream_variable)

        self.variables[variable.name] = variable

    def assemble(self):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_variables(output)
            # self.assemble_state(output)

        csdl_model = ModuleCSDL()
        for variable_name, variable in self.variables.items():
            if variable.upstream_variables is None:   # if state is an absolute input state (nothing upstream)
                num_upstream_variable_values = variable.function_space.num_coefficients
                upstream_variable_flattened_shape = (num_upstream_variable_values, 3)   # TODO: Need to figure out how to store the number of dims
                csdl_model.register_module_input(variable.name, shape=upstream_variable_flattened_shape)
            else:
                csdl_model.add_module(submodule=variable.map, name=f'{variable.name}_model')
            
        self.csdl_model = csdl_model



    def _assemble_csdl(self):
        self.assemble()

        # # Construct model group csdl model from collecting output state models
        # model_group_csdl = ModuleCSDL()
        # for output_name, output in self.outputs.items():
        #     model_group_csdl.add_module(submodule=output.absolute_map, name=f'{output_name}_model')

        # TODO: Incorporate residuals

        return self.csdl_model
