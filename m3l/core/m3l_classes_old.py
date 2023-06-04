from dataclasses import dataclass

import numpy as np
import array_mapper as am
# import scipy.sparse as sps
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

import re

@dataclass
class State:
    name : str
    # type : str
    region : list   # of manifolds
    shape : tuple   # shape of nodal values
    mesh : am.MappedArray = None   # If we are going to call this State, we need to support all states in the controls sense (so this includes cruise speed)
    # NOTE: Normally None mesh means non-spatial quantity, but for now will mean uniform mesh.
    upstream_states : list = None
    relative_map : csdl.Model = None     # Computes the State given the inputs. Can also be scipy.sparse matrix or M3L.Model
    absolute_map : csdl.Model = None
    value : np.ndarray = None   # Only has value after evaluation. Will we ever evaluate this?

@dataclass
class StateType:
    # Could potentially replace this by having specific states inherit from the State class and/or having a
    # type attribute in the State class
    type : str

# @dataclass  # why do we have this? To have special magic methods for the symbolic operations?
# class StateMap:
#     map : np.ndarray    # TODO Allow for sparse matrix maps
#     upstream_state : State = None

class Model:    # Solvers should be an instance of this

    def __init__(self, model):
        self.model = model
        self.map_in = None
        self.map_in_csdl = None
        self.map_out = None
        self.map_out_csdl = None
        self.csdl_model = None
        # attributes = vars(self.model)
        # for attribute_name, attribute_value in attributes.items():
        #     setattr(self, attribute_name, attribute_value)

    def _assemble_csdl(self, input:State, output_mesh:am.MappedArray=None):
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
    
    def evaluate_symbolic(self, input:State, output_mesh:am.MappedArray=None):
        '''
        Symbolically evaluates the model. A symbolic M3L State is returned.
        '''
        if self.csdl_model is None:
            self._assemble_csdl(input=input, output_mesh=output_mesh)

        if type(input) is State:
            num_state_values = np.cumprod(output_mesh.shape[:-1])[-1]
            output_state = State(name='solver_output', shape=output_mesh.shape, region=input.region, mesh=output_mesh, upstream_states=[input], relative_map=self.csdl_model)
            # NOTE: WARNING: the shape above will only be correct if the output is like displacement with size 3.
            return output_state

        # elif type(input) is am.MappedArray or type(input) is np.ndarray:  # could implement numerical evaluation too
        #     # I think this would ideally call the original model's evaluate method
        #     self.model.evaluate(input)




class ModelGroup:   # Implicit (or not implicit?) model groups should be an instance of this
    
    def __init__(self) -> None:
        self.models = {}
        self.states = {}
        self.outputs = {}
        self.residuals = {}
        self.parameters = None

    def add(self, submodel:Model, name:str):
        self.models[name] = submodel
        # attributes = vars(submodel)
        # for attribute_name, attribute_value in attributes.items():
        #     setattr(self, attribute_name, attribute_value)

    def register_output(self, state:State):
        self.outputs[state.name] = state

    def define_residual(self, residual:State):
        self.residuals[state.name] = residual

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

    def gather_states(self, state:State):
        if state.upstream_states is not None:
            for upstream_state in state.upstream_states:
                self.gather_states(upstream_state)

        self.states[state.name] = state

    def assemble(self):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_states(output)
            # self.assemble_state(output)

        csdl_model = ModuleCSDL()
        for state_name, state in self.states.items():
            if state.upstream_states is None:   # if state is an absolute input state (nothing upstream)
                num_upstream_state_values = np.cumprod(state.shape[:-1])[-1]
                upstream_state_flattened_shape = (num_upstream_state_values, state.shape[-1])
                csdl_model.register_module_input(state.name, shape=upstream_state_flattened_shape)
            else:
                csdl_model.add_module(submodule=state.relative_map, name=f'{state.name}_model')
            
        self.csdl_model = csdl_model



    def _assemble_csdl(self):
        self.assemble()

        # # Construct model group csdl model from collecting output state models
        # model_group_csdl = ModuleCSDL()
        # for output_name, output in self.outputs.items():
        #     model_group_csdl.add_module(submodule=output.absolute_map, name=f'{output_name}_model')

        # TODO: Incorporate residuals

        return self.csdl_model
