import m3l
import numpy as np

import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
# def map(map, input):
#     return m3l_classes.State()

def evaluate_state(state, mesh):
    num_state_values = np.cumprod(state.shape[:-1])[-1]
    temp_map = np.eye(num_state_values)
    output_name = f'nodal_{state.name}'
    relative_map = _convert_state_relative_map_to_csdl(input_state=state, linear_map=temp_map, output_name=output_name)

    new_state = m3l.State(name=output_name, shape=mesh.shape[:-1]+(state.shape[-1],), region=state.region, mesh=mesh, upstream_states=[state], relative_map=relative_map)
    return new_state

def evaluate_intrinsic_state(state, mesh):
    num_state_values = np.cumprod(state.shape[:-1])[-1]
    temp_map = np.eye(num_state_values)
    output_name = f'nodal_extrinsic_{state.name}'
    relative_map = _convert_state_relative_map_to_csdl(input_state=state, linear_map=temp_map, output_name=output_name)

    new_state = m3l.State(name=output_name, shape=mesh.shape[:-1]+(state.shape[-1],), region=state.region, mesh=mesh, upstream_states=[state], relative_map=relative_map)
    return new_state

def fit_state(state, target_state, conservative=False, model=None):
    num_state_values = np.cumprod(state.shape[:-1])[-1]
    temp_map = np.eye(num_state_values)
    output_name = f'continuous_{state.name}'
    relative_map = _convert_state_relative_map_to_csdl(input_state=state, linear_map=temp_map, output_name=output_name)

    new_state = m3l.State(name=output_name, shape=target_state.shape[:-1], region=target_state.region, mesh=target_state.mesh, upstream_states=[state], relative_map=relative_map)
    return new_state

def fit_intrinsic_state(state, target_state, conservative=False, model=None):
    num_state_values = np.cumprod(state.shape[:-1])[-1]
    temp_map = np.eye(num_state_values)
    output_name = f'continuous_intrinsic_{state.name}'
    relative_map = _convert_state_relative_map_to_csdl(input_state=state, linear_map=temp_map, output_name=output_name)

    new_state = m3l.State(name=output_name, shape=target_state.shape[:-1], region=target_state.region, mesh=target_state.mesh, upstream_states=[state], relative_map=relative_map)
    return new_state

def _convert_state_relative_map_to_csdl(input_state, linear_map, output_name):
    relative_map = ModuleCSDL()
    num_input_state_values = np.cumprod(input_state.shape[:-1])[-1]
    input_state_flattened_shape = (num_input_state_values,input_state.shape[-1])
    input_state_csdl = relative_map.register_module_input(input_state.name, shape=input_state_flattened_shape)
    linear_map_csdl = relative_map.create_input(f'{output_name}_map_csdl', linear_map)
    output_state = csdl.matmat(linear_map_csdl, input_state_csdl)
    relative_map.register_module_output(output_name, output_state)
    return relative_map

# def convert_map_to_csdl(map, input_name, output_name):
    # pass