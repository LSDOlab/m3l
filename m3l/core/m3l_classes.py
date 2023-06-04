from dataclasses import dataclass
from typing import Any

import numpy as np
import array_mapper as am
# import scipy.sparse as sps
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

import re

''' Classes representing spaces '''
@dataclass
class Field:
    pass    # This may potentially have more information in the future

@dataclass
class VectorSpace:
    field : Field
    dimensions : tuple

@dataclass
class FunctionSpace:
    # reference_geometry : Function = None
    pass    # do we want separate class for state functions that point to a reference geometry?


''' Classes functions and information '''
@dataclass(kw_only=True)
class Variable:
    name : str
    upstream_variables : dict = None
    map : csdl.Model = None
    value : np.ndarray = None

@dataclass(kw_only=True)
class Vector(Variable):
    values : np.ndarray
    # value : np.ndarray = values

    def __post_init__(self):
        self.value = self.values

@dataclass(kw_only=True)
class FunctionValues(Variable):
    mesh : am.MappedArray
    values : np.ndarray = None
    # value : np.ndarray = values

    def __post_init__(self):
        self.value = self.values

@dataclass(kw_only=True)
class Function(Variable):
    function_space : FunctionSpace
    # coefficients: int = field(default_factory=lambda: None, metadata={"orig_name": "original_attribute"})
    coefficients : np.ndarray = None
    # value : np.ndarray = coefficients   # not sure I actually want this.

    def __call__(self, mesh) -> FunctionValues:
        return self.evaluate(mesh)

    def evaluate(self, mesh):
        num_values = np.prod(mesh.shape[:-1])
        temp_map = np.eye(num_values, self.function_space.num_coefficients)
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
class Model:    # Solvers should be an instance of this

    # def __init__(self, model):
    #     self.model = model
    #     self.map_in = None
    #     self.map_in_csdl = None
    #     self.map_out = None
    #     self.map_out_csdl = None
    #     self.csdl_model = None
    #     # attributes = vars(self.model)
    #     # for attribute_name, attribute_value in attributes.items():
    #     #     setattr(self, attribute_name, attribute_value)

    def _assemble_csdl(self, input:Function, output_mesh:am.MappedArray=None):
        model_csdl = self.model._assemble_csdl()
        self.model.construct_map_in(input)
        self.model.construct_map_out(output_mesh)

        model_name_camel = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', type(model_csdl).__name__)
        model_name_snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', model_name_camel).lower()
        map_in_csdl = self.model.map_in_csdl
        # if map_in_csdl is None:
        #     pass
        #     map_in_csdl = identity
        map_out_csdl = self.model.map_out_csdl
        # if map_in_csdl is None:
        #     map_out_csdl = identity

        csdl_model = ModuleCSDL()
        csdl_model.add(map_in_csdl, 'map_in')
        csdl_model.add_module(model_csdl, model_name_snake)
        csdl_model.add(map_out_csdl, 'map_out')
        self.csdl_model = csdl_model

        return csdl_model
    
    # def evaluate(self, model_map=csdl.Model, inputs:list, outputs:list):
        
    #     input_mappings_csdl = csdl.Model()
    #     for input, input_map in inputs.items():
    #         if input is not None:
    #             num_state = np.prod(input.mesh.shape[:-1])
    #             nodal_forces = input_mappings_csdl.declare_variable(name=input.name, shape=(num_state,input.num_dimensions))
    #             model_force_inputs = csdl.matmat(input_map, nodal_forces)
    #             input_mappings_csdl.register_output(, model_force_inputs)




    #     if nodal_forces is not None:
    #         num_forces = np.cumprod(nodal_forces.shape[:-1])[-1]
    #         nodal_forces = input_mappings_csdl.declare_variable(name='nodal_forces', shape=(num_forces,nodal_forces.shape[-1]))
    #         model_force_inputs = csdl.matmat(force_map, nodal_forces)
    #         input_mappings_csdl.register_output('left_wing_beam_forces', model_force_inputs)
    #     if nodal_moments is not None:
    #         num_moments = np.cumprod(nodal_moments.shape[:-1])[-1]
    #         nodal_moments = input_mappings_csdl.declare_variable(name='nodal_moments', shape=(num_moments,nodal_moments.shape[-1]))
    #         model_moment_inputs = csdl.matmat(moment_map, nodal_moments)
    #         input_mappings_csdl.register_output('left_wing_beam_moments', model_moment_inputs)

    #     beam_csdl = self._assemble_csdl()

    #     output_mappings_csdl = csdl.Model()
    #     if nodal_displacements is not None:
    #         num_displacements = np.cumprod(nodal_displacements.shape[:-1])[-1]
    #         nodal_displacements = output_mappings_csdl.declare_variable(name='left_wing_beam_displacements', shape=(num_displacements,nodal_displacements.shape[-1]))
    #         model_displacement_outputs = csdl.matmat(displacement_map, nodal_displacements)
    #         output_mappings_csdl.register_output('nodal_displacements', model_displacement_outputs)
    #     if nodal_rotations is not None:
    #         num_rotations = np.cumprod(nodal_rotations.shape[:-1])[-1]
    #         nodal_rotations = output_mappings_csdl.declare_variable(name='left_wing_beam_rotations', shape=(num_rotations,nodal_rotations.shape[-1]))
    #         model_rotation_outputs = csdl.matmat(rotation_map, nodal_rotations)
    #         output_mappings_csdl.register_output('nodal_rotations', model_rotation_outputs)

    #     csdl_model.add(submodel=input_mappings_csdl, name='beam_inputs_mapping')
    #     csdl_model.add(submodel=beam_csdl, name='beam_model')
    #     csdl_model.add(submodel=output_mappings_csdl, name='beam_outputs_mapping')

    #     nodal_displacements = m3l.NodalState(mesh=nodal_outputs_mesh, upstream_states=[nodal_forces, nodal_moments], map=csdl_model)
    #     nodal_rotations = m3l.NodalState(mesh=nodal_outputs_mesh, upstream_states=[nodal_forces, nodal_moments], map=csdl_model)
    
    def evaluate_symbolic(self, input:Function, output_mesh:am.MappedArray=None):
        '''
        Symbolically evaluates the model. A symbolic M3L Function is returned.
        '''
        if self.csdl_model is None:
            self._assemble_csdl(input=input, output_mesh=output_mesh)

        if type(input) is Function:
            num_state_values = np.cumprod(output_mesh.shape[:-1])[-1]
            output_state = Function(name='solver_output', shape=output_mesh.shape, region=input.region, mesh=output_mesh, upstream_states=[input], relative_map=self.csdl_model)
            # NOTE: WARNING: the shape above will only be correct if the output is like displacement with size 3.
            return output_state

        # elif type(input) is am.MappedArray or type(input) is np.ndarray:  # could implement numerical evaluation too
        #     # I think this would ideally call the original model's evaluate method
        #     self.model.evaluate(input)




class ModelGroup:   # Implicit (or not implicit?) model groups should be an instance of this
    
    def __init__(self) -> None:
        self.models = {}
        self.variables = {}
        self.outputs = {}
        self.residuals = {}
        self.parameters = None

    def add(self, submodel:Model, name:str):
        self.models[name] = submodel
        # attributes = vars(submodel)
        # for attribute_name, attribute_value in attributes.items():
        #     setattr(self, attribute_name, attribute_value)

    def add_output(self, output:Function):
        self.outputs[output.name] = output

    def add_residual(self, residual:Function):
        self.residuals[residual.name] = residual

    def set_implicit_solver(self, solver):
        self.implicit_solver = solver

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
