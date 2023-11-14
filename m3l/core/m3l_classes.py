from dataclasses import dataclass
from typing import Any, List

import numpy as np
import scipy.sparse as sps
from scipy import linalg
import array_mapper as am
# import scipy.sparse as sps
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module

from m3l.core.csdl_operations import Eig, EigExplicit

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

class VStack(ExplicitOperation):
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
        # shape = x1.shape
        # shape[0] = x2.shape[0]
        shape = self.shape
        output_name = f'{x1.name}_stack_{x2.name}'
        operation_csdl = csdl.Model()
        x1_csdl = operation_csdl.declare_variable(name='x1', shape=x1.shape)
        x2_csdl = operation_csdl.declare_variable(name='x2', shape=x2.shape)
        y = operation_csdl.create_output(name=output_name, shape=shape)
        if len(shape) == 1:
            y[0:x1.shape[0]] = x1_csdl
            y[x1.shape[0]:] = x2_csdl
        elif len(shape) == 2:
            y[0:x1.shape[0],:] = x1_csdl
            y[x1.shape[0]:,:] = x2_csdl
        elif len(shape) == 3:
            y[0:x1.shape[0],:,:] = x1_csdl
            y[x1.shape[0]:,:,:] = x2_csdl
        else:
            raise Exception('VStack not implemented for tensors of dim 4 or greator')
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

    def evaluate(self, x1:Variable, x2:Variable, design_condition=None) -> Variable:
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
        if design_condition:
            dc_name = design_condition.parameters['name']
            self.name = f'{dc_name}_{x1.name}_stack_{x2.name}_operation'
        else:
            self.name = f'{x1.name}_stack_{x2.name}_operation'

        # Define operation arguments
        self.arguments = {'x1' : x1, 'x2' : x2}
        # shape = x1.shape
        # shape[0] = x2.shape[0]

        self.shape = (x1.shape[0] + x2.shape[0], ) + x1.shape[1:]
        # exit(shape)
        # Create the M3L variables that are being output
        function_values = Variable(name=f'{x1.name}_stack_{x2.name}', shape=self.shape, operation=self)
        return function_values

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
class IndexedFunctionSpace:
    name : str
    spaces : dict[str, FunctionSpace]

    def compute_evaluation_map(self, indexed_parametric_coordinates:list[tuple[str, np.ndarray]]) -> list:
        # TODO: use agrigated knot vectors. For now, we have this dumb loop:
        map = []
        for item in indexed_parametric_coordinates:
            space = self.spaces[item[0]]
            coords = self.spaces[item[1]]
            map_i = space.compute_evaluation_map(coords)
            map.append(map_i)
        return map

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

@dataclass
class IndexedFunction:
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
    space : IndexedFunctionSpace
    coefficients : dict[str, Variable]

    def __call__(self, mesh : am.MappedArray) -> Variable:
        return self.evaluate(mesh)

    def evaluate(self, indexed_parametric_coordinates) -> Variable:
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
        function_evaluation_model = IndexedFunctionEvaluation(function=self, indexed_parametric_coordinates=indexed_parametric_coordinates)
        function_values = function_evaluation_model.evaluate()
        return function_values

    def evaluate_conservative(self, 
                              fitting_matrix_key_list, 
                              num_oml_output_points=None,
                              num_oml_dual_variable_points=None,
                              composition_mat=None,
                              otherside_composed_mat=None, 
                              solver_invariant_mat=None, 
                              framework_invariant_mat=None) -> Variable:
        '''
        Construct evaluation map such that it conserves aeroelastic work.

        Parameters
        ----------
        mesh : am.MappedArray
            The mesh to evaluate over.

        Returns
        -------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        # TODO: Compute projection matrix based on invariant matrices and other projection matrices, then distribute the
        # Compute projection matrix
        
        # incoming_projection_T_times_solver_mat = composition_mat.T@solver_invariant_mat
        # pinv_incoming_projection_T_times_solver_mat = np.linalg.pinv(incoming_projection_T_times_solver_mat.toarray())
        # framework_mat_times_otherside_composition = framework_invariant_mat@otherside_composed_mat

        # # inv_term = incoming_projection_T_times_solver_mat@pinv_incoming_projection_T_times_solver_mat

        # # TODO: Matrix below could be made more sparse, by switching the inverse-distance weighting to a function with a compact support
        # # construct the projection matrix itself
        # proj_mat_T = framework_mat_times_otherside_composition@pinv_incoming_projection_T_times_solver_mat
        # proj_mat = proj_mat_T.T

        # # now that we have the projection matrix we construct a vector of the inputs by looping over the provided keys
        # coefficient_obj_list = []
        # for key in fitting_matrix_key_list:
        #     coefficient_obj = self.coefficients[key]
        #     surf_coefficient_arr = coefficient_obj.value.reshape((-1, coefficient_obj.shape[-1]))

        #     coefficient_obj_list += [surf_coefficient_arr]

        # # make vertical stack of coefficient objects
        # coefficients_stacked = np.vstack(coefficient_obj_list)
        # # stack the [x, y, z] components of the coefficients
        # coefficients_vec_flat = coefficients_stacked.flatten(order='F')

        # # multiply projection matrix with flat coefficient matrix
        # output_vec = proj_mat@coefficients_vec_flat

        # TODO: Add input argument that encodes the order of the surfaces, so we can match the right projection columns to the right surface coefficients

        conservative_function_evaluation_model = IndexedFunctionConservativeEvaluation(function=self, 
                                                                                       vector_surface_order=fitting_matrix_key_list,
                                                                                       solver_invariant_mat=solver_invariant_mat,
                                                                                       framework_invariant_mat=framework_invariant_mat,
                                                                                       num_oml_output_points=num_oml_output_points,
                                                                                       num_oml_dual_variable_points=num_oml_dual_variable_points
                                                                                       )
        function_values = conservative_function_evaluation_model.evaluate(composition_mat, otherside_composed_mat)
        return function_values
    
    def inverse_evaluate(self, indexed_parametric_coordinates, function_values:Variable, regularization_coeff:float=None):
        '''
        Performs an inverse evaluation to set the coefficients of this function given an input of evaluated points over a mesh.

        Parameters
        ----------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        # Perform B-spline fit 
        inverse_operation = IndexedFunctionInverseEvaluation(function=self, indexed_parametric_coordinates=indexed_parametric_coordinates, regularization_coeff=regularization_coeff)
        inverse_operation.evaluate(function_values=function_values)
        for key, value in self.coefficients.items():
            value.operation = inverse_operation
        return self.coefficients
    
    def evaluate_normals(self, indexed_parametric_coordinates, name:str=None):
        function_evaluation_model = IndexedFunctionNormalEvaluation(function=self, indexed_parametric_coordinates=indexed_parametric_coordinates, name=name)
        function_values = function_evaluation_model.evaluate()
        return function_values
    
    def compute(self, indexed_parametric_coordinates, coefficients):
        associated_coords = {}
        index = 0
        for item in indexed_parametric_coordinates:
            key = item[0]
            value = item[1]
            if key not in associated_coords.keys():
                associated_coords[key] = [[index], value]
            else:
                associated_coords[key][0].append(index)
                associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
            index += 1

        output_shape = (len(indexed_parametric_coordinates), coefficients[indexed_parametric_coordinates[0][0]].shape[-1])

        evaluated_points = np.zeros(output_shape)
        for key, value in associated_coords.items(): # in the future, use submodels from the function spaces?
            evaluation_matrix = self.space.spaces[key].compute_evaluation_map(value[1])
            evaluated_points[value[0],:] = evaluation_matrix.dot(coefficients[key].reshape((-1, coefficients[key].shape[-1])))
        return evaluated_points


class IndexedFunctionConservativeEvaluation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('function', types=IndexedFunction)
        self.parameters.declare('vector_surface_order', types=list)
        self.parameters.declare('solver_invariant_mat', types=sps.coo_matrix)
        self.parameters.declare('framework_invariant_mat', types=sps.coo_matrix)
        self.parameters.declare('num_oml_output_points', types=int)
        self.parameters.declare('num_oml_dual_variable_points', types=int)

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.function = self.parameters['function']
        self.vector_surface_order = self.parameters['vector_surface_order']
        self.solver_invariant_mat = self.parameters['solver_invariant_mat']
        self.framework_invariant_mat = self.parameters['framework_invariant_mat']
        self.num_oml_output_points = self.parameters['num_oml_output_points']
        self.num_oml_dual_variable_points = self.parameters['num_oml_dual_variable_points']

    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        csdl_map = ModuleCSDL()

        # TODO: Conditioning of `self.composition_mat` below is extremely poor (4.6e17) and breaks down the computations
        #       This is likely because of the inverse distance weighting approach -- add a fix for this by removing extremely small numbers
        incoming_projection_T_times_solver_mat = self.composition_mat.T@self.solver_invariant_mat
        # repeat matrix along diagonal three times and compute pseudo-inverse
        pinv_incoming_projection_T_times_solver_mat = np.linalg.pinv(sps.block_diag([incoming_projection_T_times_solver_mat]*3).toarray())
        framework_mat_times_otherside_composition = self.framework_invariant_mat@self.otherside_composed_mat
        framework_mat_times_otherside_composition_stacked = sps.block_diag([framework_mat_times_otherside_composition]*3)

        # inv_term = incoming_projection_T_times_solver_mat@pinv_incoming_projection_T_times_solver_mat

        # TODO: Matrix below could be made more sparse, by switching the inverse-distance weighting to a function with a compact support
        # construct the projection matrix itself
        proj_mat_T = framework_mat_times_otherside_composition_stacked@pinv_incoming_projection_T_times_solver_mat
        proj_mat = proj_mat_T.T

        proj_mat_csdl = csdl_map.register_module_input('conservative_projection_matrix', shape=proj_mat.shape,
                                                                    val=proj_mat)

        # after having constructed the projection matrix we construct the vector that will contain the framework coefficients
        
        coeff_arr_list = []
        coefficients_cumsum_per_coordinate = [0]
        for surf_name in self.vector_surface_order:
            coeff_arr = self.function.coefficients[surf_name]
            coeff_arr_reshaped = coeff_arr.value.reshape((-1, coeff_arr.shape[-1]))
            coeff_arr_reshaped_csdl = csdl_map.register_module_input(coeff_arr.name, shape=coeff_arr_reshaped.shape,
                                                                    val=coeff_arr_reshaped)
            coeff_arr_list += [coeff_arr_reshaped_csdl]
            coefficients_cumsum_per_coordinate += [coefficients_cumsum_per_coordinate[-1] + coeff_arr_reshaped.shape[0]]
        
        # create CSDL array that contains all coefficients
        coefficients_concatenated_csdl = csdl_map.create_output(name='inp_coefficients_concatenated',shape=(coefficients_cumsum_per_coordinate[-1], 3))
        for i in range(len(coeff_arr_list)):
            coefficients_concatenated_csdl[coefficients_cumsum_per_coordinate[i]:coefficients_cumsum_per_coordinate[i+1], :] = coeff_arr_list[i]
        
        # next we reshape the concatenated coefficient array by stacking its columns on top of one another
        coefficients_concatenated_csdl_flat = csdl.reshape(coefficients_concatenated_csdl, (coefficients_cumsum_per_coordinate[-1]*3,))

        # Now we FINALLY multiply the projection matrix with the flattened coefficient array
        output_oml_coefficients = csdl.matvec(proj_mat_csdl, coefficients_concatenated_csdl_flat)
        output_oml_coefficients_reshaped = csdl.reshape(output_oml_coefficients, (self.num_oml_output_points, 3))
        output_map = csdl_map.create_output(f'evaluated_{self.function.name}', shape=output_oml_coefficients_reshaped.shape)
        output_map[:, :] = output_oml_coefficients_reshaped

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

    def evaluate(self, composition_mat, otherside_composed_mat):
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
        # surface_names = []
        # for item in self.indexed_mesh:
        #     name = item[0]
        #     if not name in surface_names:
        #         surface_names.append(name)
        self.arguments = {}
        coefficients = self.function.coefficients
        for name in self.vector_surface_order:
            self.arguments[coefficients[name].name] = coefficients[name]
        self.arguments = self.function.coefficients

        self.solver_invariant_mat_stacked = sps.block_diag([self.solver_invariant_mat]*3)
        self.framework_invariant_mat_stacked = sps.block_diag([self.framework_invariant_mat]*3)
        self.composition_mat = composition_mat
        self.otherside_composed_mat = otherside_composed_mat

        # Create the M3L variables that are being output
        output_shape = (self.num_oml_output_points, self.function.coefficients[self.vector_surface_order[0]].shape[-1])

        function_values = Variable(name=f'evaluated_{self.function.name}', shape=output_shape, operation=self)
        return function_values

    # def compute(self):
    #     '''
    #     Creates the CSDL model to compute the function evaluation.

    #     Returns
    #     -------
    #     csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
    #         The csdl model or module that computes the model/operation outputs.
    #     '''
    #     csdl_map = ModuleCSDL()

    #     # register the solver and framework invariant matrices
    #     solver_invariant_mat = csdl_map.register_module_input(name='solver_invariant_mat', 
    #                                                        val=self.solver_invariant_mat.toarray())
    #     framework_invariant_mat = csdl_map.register_module_input(name='framework_invariant_mat', 
    #                                                        val=self.framework_invariant_mat.toarray())        

    #     # import oml -> solver map (that is applied directly after the current map)
    #     oml_to_solver_map = csdl_map.register_module_input(name='wing_displacement_input_function_displacements_map', 
    #                                                        shape=(self.solver_invariant_mat.shape[1], self.num_oml_output_points))

    #     # import solver -> oml map on the other side of the aeroelastic coupling diagram
    #     solver_to_oml_map = csdl_map.register_module_input(name='wing_force_map', shape=(self.num_oml_dual_variable_points, self.solver_invariant_mat.shape[0]))

    #     # import oml -> framework map on the other side of the aeroelastic coupling diagram
    #     oml_to_framework_map_list = []
    #     total_oml_to_framework_map_rows = 0
    #     for surf_name in self.vector_surface_order:
    #         # TODO: Replace the shape of the `self.function` (displacement) term below with the number of points that are used to represent the force on the framework
    #         total_oml_to_framework_map_rows += np.prod(self.function.coefficients[surf_name].shape[:-1])
    #         oml_to_framework_map = csdl_map.register_module_input(name='fitting_matrix_{}'.format(surf_name), shape=(np.prod(self.function.coefficients[surf_name].shape[:-1]), self.num_oml_dual_variable_points))
    #         oml_to_framework_map_list += [oml_to_framework_map]

    #     # we combine all `oml_to_framework_map` instances into a single big array
    #     oml_to_framework_maps_all_surfaces = csdl_map.create_output(name='oml_to_framework_maps_all_surfaces', shape=(total_oml_to_framework_map_rows, self.num_oml_dual_variable_points))
    #     current_insert_start = 0
    #     current_insert_end = 0  # oml_to_framework_map_list[0].shape[0]
    #     for oml_to_framework_object in oml_to_framework_map_list:
    #         current_insert_end += oml_to_framework_object.shape[0]
    #         oml_to_framework_maps_all_surfaces[current_insert_start:current_insert_end, :] = oml_to_framework_object
    #         current_insert_start += oml_to_framework_object.shape[0]

    #     # Now that we have all three of the component-specific maps we stack each one three times along the diagonal of larger matrices,
    #     # so that later on we can directly matvec multiply the [x; y; z] input vector to obtain the corresponding output
        
    #     # First we define the stacked csdl objects
    #     # oml_to_solver_map_stacked = csdl_map.create_output(name='oml_to_solver_map_stacked', shape=(3*self.solver_invariant_mat.shape[0], 3*self.num_oml_output_points))
    #     # solver_to_oml_map_stacked = csdl_map.create_output(name='solver_to_oml_map_stacked', shape=(3*self.num_oml_dual_variable_points, 3*self.solver_invariant_mat.shape[1]))
    #     # oml_to_framework_maps_all_surfaces_stacked = csdl_map.create_output(name='oml_to_framework_maps_all_surfaces_stacked', shape=(3*total_oml_to_framework_map_rows, 3*self.num_oml_dual_variable_points))

    #     # loop over all three principal directions (x, y, z) and fill the corresponding diagonal blocks each time
    #     # for i in range(3):
    #         # oml_to_solver_map_stacked[i*self.solver_invariant_mat.shape[0]:(i+1)*self.solver_invariant_mat.shape[0], i*self.num_oml_output_points:(i+1)*self.num_oml_output_points] = 1*oml_to_solver_map
    #         # solver_to_oml_map_stacked[i*self.num_oml_dual_variable_points:(i+1)*self.num_oml_dual_variable_points, i*self.solver_invariant_mat.shape[1]:(i+1)*self.solver_invariant_mat.shape[1]] = 1*solver_to_oml_map
    #         # oml_to_framework_maps_all_surfaces_stacked[i*total_oml_to_framework_map_rows:(i+1)*total_oml_to_framework_map_rows, i*self.num_oml_dual_variable_points:(i+1)*self.num_oml_dual_variable_points] = 1*oml_to_framework_maps_all_surfaces
    #         # oml_to_solver_map_mult = 1*oml_to_solver_map_mult

    #     # Now we have all three stacked maps and the stacked invariant matrices are stored in `self`, so we can compute the work-conserving projection map
    #     otherside_composition_map = csdl.matmat(oml_to_framework_maps_all_surfaces, solver_to_oml_map)
    #     framework_mat_times_otherside_composition = csdl.matmat(framework_invariant_mat, otherside_composition_map)

    #     # we stack the matrix above three times along the diagonal
    #     framework_mat_times_otherside_composition_stacked = csdl_map.create_output(name='framework_mat_times_otherside_composition_stacked', shape=(3*framework_mat_times_otherside_composition.shape[0], 3*framework_mat_times_otherside_composition.shape[1]))
    #     for i in range(3):
    #         framework_mat_times_otherside_composition_stacked[i*framework_mat_times_otherside_composition.shape[0]:(i+1)*framework_mat_times_otherside_composition.shape[0], i*framework_mat_times_otherside_composition.shape[1]:(i+1)*framework_mat_times_otherside_composition.shape[1]] = 1*framework_mat_times_otherside_composition

    #     # we hack our way to computing the pseudo-inverse of the product of two matrices 
    #     oml_to_solver_map_transpose = csdl.transpose(oml_to_solver_map)
    #     oml_to_solver_map_T_times_solver_mat = csdl.matmat(oml_to_solver_map_transpose, solver_invariant_mat)

    #     # we repeat the matrix three times along the diagonal, compute the pseudo-inverse with Numpy and then convert it back to a CSDL object
    #     pinv_oml_to_solver_map_T_times_solver_mat = np.linalg.pinv(sps.block_diag([oml_to_solver_map_T_times_solver_mat.val]*3).toarray())
    #     pinv_oml_to_solver_map_T_times_solver_mat_csdl = csdl_map.create_input(name='pinv_oml_to_solver_map_T_times_solver_mat', val=pinv_oml_to_solver_map_T_times_solver_mat)

    #     # Now we can finally construct the actual projection matrix
    #     proj_mat_transpose = csdl.matmat(framework_mat_times_otherside_composition_stacked, pinv_oml_to_solver_map_T_times_solver_mat_csdl)
    #     proj_mat = csdl.transpose(proj_mat_transpose)

    #     # after having constructed the projection matrix we construct the vector that will contain the framework coefficients
        
    #     coeff_arr_list = []
    #     coefficients_cumsum_per_coordinate = [0]
    #     for surf_name in self.vector_surface_order:
    #         coeff_arr = self.function.coefficients[surf_name]
    #         coeff_arr_reshaped = coeff_arr.value.reshape((-1, coeff_arr.shape[-1]))
    #         coeff_arr_reshaped_csdl = csdl_map.register_module_input(coeff_arr.name, shape=coeff_arr_reshaped.shape,
    #                                                                 val=coeff_arr_reshaped)
    #         coeff_arr_list += [coeff_arr_reshaped_csdl]
    #         coefficients_cumsum_per_coordinate += [coefficients_cumsum_per_coordinate[-1] + coeff_arr_reshaped.shape[0]]
        
    #     # create CSDL array that contains all coefficients
    #     coefficients_concatenated_csdl = csdl_map.create_output(name='inp_coefficients_concatenated',shape=(coefficients_cumsum_per_coordinate[-1], 3))
    #     for i in range(len(coeff_arr_list)):
    #         coefficients_concatenated_csdl[coefficients_cumsum_per_coordinate[i]:coefficients_cumsum_per_coordinate[i+1], :] = coeff_arr_list[i]
        
    #     # next we reshape the concatenated coefficient array by stacking its columns on top of one another
    #     coefficients_concatenated_csdl_flat = csdl.reshape(coefficients_concatenated_csdl, (coefficients_cumsum_per_coordinate[-1]*3,))

    #     # Now we FINALLY multiply the projection matrix with the flattened coefficient array
    #     output_oml_coefficients = csdl.matvec(proj_mat, coefficients_concatenated_csdl_flat)
    #     output_oml_coefficients_reshaped = csdl.reshape(output_oml_coefficients, (self.num_oml_output_points, 3))
    #     output_map = csdl_map.create_output(f'evaluated_{self.function.name}', shape=output_oml_coefficients_reshaped.shape)
    #     output_map[:, :] = output_oml_coefficients_reshaped

    #     return csdl_map
    
    # def compute_derivates(self):
    #     '''
    #     -- optional --
    #     Creates the CSDL model to compute the derivatives of the model outputs. This is only needed for dynamic analysis.
    #     For now, I would recommend coming back to this.

    #     Returns
    #     -------
    #     derivatives_csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
    #         The csdl model or module that computes the derivatives of the model/operation outputs.
    #     '''
    #     pass

    # def evaluate(self, num_oml_output_points):
    #     '''
    #     User-facing method that the user will call to define a model evaluation.

    #     Parameters
    #     ----------
    #     mesh : Variable
    #         The mesh over which the function will be evaluated.

    #     Returns
    #     -------
    #     function_values : Variable
    #         The values of the function at the mesh locations.
    #     '''
    #     self.name = f'{self.function.name}_evaluation'

    #     # Define operation arguments
    #     # surface_names = []
    #     # for item in self.indexed_mesh:
    #     #     name = item[0]
    #     #     if not name in surface_names:
    #     #         surface_names.append(name)
    #     self.arguments = {}
    #     coefficients = self.function.coefficients
    #     for name in self.vector_surface_order:
    #         self.arguments[coefficients[name].name] = coefficients[name]
    #     self.arguments = self.function.coefficients

    #     self.solver_invariant_mat_stacked = sps.block_diag([self.solver_invariant_mat]*3)
    #     self.framework_invariant_mat_stacked = sps.block_diag([self.framework_invariant_mat]*3)

    #     # Create the M3L variables that are being output
    #     output_shape = (num_oml_output_points, self.function.coefficients[self.vector_surface_order[0]].shape[-1])

    #     function_values = Variable(name=f'evaluated_{self.function.name}', shape=output_shape, operation=self)
    #     return function_values


class IndexedFunctionEvaluation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('function', types=IndexedFunction)
        self.parameters.declare('indexed_parametric_coordinates', types=list)

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.function = self.parameters['function']
        self.indexed_mesh = self.parameters['indexed_parametric_coordinates']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''

        associated_coords = {}
        index = 0
        for item in self.indexed_mesh:
            key = item[0]
            value = item[1]
            if key not in associated_coords.keys():
                associated_coords[key] = [[index], value]
            else:
                associated_coords[key][0].append(index)
                associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
            index += 1

        output_name = f'evaluated_{self.function.name}'
        output_shape = (len(self.indexed_mesh), self.function.coefficients[self.indexed_mesh[0][0]].shape[-1])
        csdl_map = ModuleCSDL()
        points = csdl_map.create_output(output_name, shape=output_shape)
        
        coefficients_csdl = {} 
        for key, coefficients in self.function.coefficients.items():
            num_coefficients = np.prod(coefficients.shape[:-1])
            if coefficients.value is None:
                coefficients_csdl[key] = csdl_map.register_module_input(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]))
            else:
                coefficients_csdl[key] = csdl_map.register_module_input(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
                                                                    val=coefficients.value.reshape((-1, coefficients.shape[-1])))

        for key, value in associated_coords.items():
            evaluation_matrix = self.function.space.spaces[key].compute_evaluation_map(value[1])
            if sps.issparse(evaluation_matrix):
                evaluation_matrix = evaluation_matrix.toarray()
            evaluation_matrix_csdl = csdl_map.register_module_input('evaluation_matrix_'+key, val=evaluation_matrix, shape = evaluation_matrix.shape, computed_upstream=False)
            associated_function_values = csdl.matmat(evaluation_matrix_csdl, coefficients_csdl[key])
            for i in range(len(value[0])):
                points[value[0][i],:] = associated_function_values[i,:]

        # unique_keys = []
        # for item in self.indexed_mesh:
        #     if not item[0] in unique_keys:
        #         unique_keys.append(item[0])
        #     map = self.function.space.spaces[item[0]].compute_evaluation_map(item[1])
        #     if sps.issparse(map):
        #         map = map.toarray()
        #     map_csdl = csdl_map.create_input(f'{self.name}_evaluation_map_{str(index)}', map)
        #     function_coefficients = coefficients_csdl[item[0]]
        #     flattened_point = csdl.matmat(map_csdl, function_coefficients)
        #     new_shape = (1,output_shape[-1])
        #     point = csdl.reshape(flattened_point, new_shape=new_shape)
        #     points[index,:] = point
        #     index += 1
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

    def evaluate(self):
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
        surface_names = []
        for item in self.indexed_mesh:
            name = item[0]
            if not name in surface_names:
                surface_names.append(name)
        self.arguments = {}
        coefficients = self.function.coefficients
        for name in surface_names:
            self.arguments[coefficients[name].name] = coefficients[name]
        # self.arguments = self.function.coefficients

        # Create the M3L variables that are being output
        output_shape = (len(self.indexed_mesh), self.function.coefficients[self.indexed_mesh[0][0]].shape[-1])

        function_values = Variable(name=f'evaluated_{self.function.name}', shape=output_shape, operation=self)
        return function_values
    

class IndexedFunctionNormalEvaluation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('function', types=IndexedFunction)
        self.parameters.declare('indexed_parametric_coordinates', types=list)
        self.parameters.declare('name', types=str, allow_none=True)

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.function = self.parameters['function']
        self.indexed_mesh = self.parameters['indexed_parametric_coordinates']
        self.input_name = self.parameters['name']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''

        associated_coords = {}
        index = 0
        for item in self.indexed_mesh:
            key = item[0]
            value = item[1]
            if key not in associated_coords.keys():
                associated_coords[key] = [[index], value]
            else:
                associated_coords[key][0].append(index)
                associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
            index += 1

        output_name = f'evaluated_normal_{self.function.name}'
        if not self.input_name is None:
            output_name = output_name + '_' + self.input_name
        output_shape = (len(self.indexed_mesh), self.function.coefficients[self.indexed_mesh[0][0]].shape[-1])
        csdl_map = ModuleCSDL()
        points = csdl_map.create_output(output_name, shape=output_shape)
        
        coefficients_csdl = {} 
        for key, coefficients in self.function.coefficients.items():
            num_coefficients = np.prod(coefficients.shape[:-1])
            if coefficients.value is None:
                coefficients_csdl[key] = csdl_map.register_module_input(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]))
            else:
                coefficients_csdl[key] = csdl_map.register_module_input(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
                                                                    val=coefficients.value.reshape((-1, coefficients.shape[-1])))

        for key, value in associated_coords.items():
            evaluation_matrix_u = self.function.space.spaces[key].compute_evaluation_map(value[1], parametric_derivative_order=(1,0))
            evaluation_matrix_v = self.function.space.spaces[key].compute_evaluation_map(value[1], parametric_derivative_order=(0,1))
            if sps.issparse(evaluation_matrix_u):
                evaluation_matrix_u = evaluation_matrix_u.toarray()
                evaluation_matrix_v = evaluation_matrix_v.toarray()

            evaluation_matrix_u_csdl = csdl_map.register_module_input('evaluation_matrix_u_'+key, val=evaluation_matrix_u, shape = evaluation_matrix_u.shape, computed_upstream=False)
            evaluation_matrix_v_csdl = csdl_map.register_module_input('evaluation_matrix_v_'+key, val=evaluation_matrix_v, shape = evaluation_matrix_v.shape, computed_upstream=False)

            associated_u_function_values = csdl.matmat(evaluation_matrix_u_csdl, coefficients_csdl[key])
            associated_v_function_values = csdl.matmat(evaluation_matrix_v_csdl, coefficients_csdl[key])

            normals = csdl.cross(associated_u_function_values, associated_v_function_values, axis=1)
            normals = normals / csdl.expand(csdl.pnorm(normals, axis=1), normals.shape, 'i->ij')

            for i in range(len(value[0])):
                points[value[0][i],:] = normals[i,:]

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

    def evaluate(self):
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
        if self.input_name is not None:
            self.name = f'{self.function.name}_normal_evaluation_' + self.input_name
        else:
            self.name = f'{self.function.name}_normal_evaluation'

        # Define operation arguments
        surface_names = []
        for item in self.indexed_mesh:
            name = item[0]
            if not name in surface_names:
                surface_names.append(name)
        self.arguments = {}
        coefficients = self.function.coefficients
        for name in surface_names:
            self.arguments[coefficients[name].name] = coefficients[name]
        # self.arguments = self.function.coefficients

        # Create the M3L variables that are being output
        output_shape = (len(self.indexed_mesh), self.function.coefficients[self.indexed_mesh[0][0]].shape[-1])

        output_name = f'evaluated_normal_{self.function.name}'
        if not self.input_name is None:
            output_name = output_name + '_' + self.input_name

        function_values = Variable(name=output_name, shape=output_shape, operation=self)
        return function_values

class IndexedFunctionInverseEvaluation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('function', types=IndexedFunction)
        self.parameters.declare('indexed_parametric_coordinates', types=list)
        self.parameters.declare('function_values')
        self.parameters.declare('regularization_coeff', default=None)

    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.function = self.parameters['function']
        self.indexed_mesh = self.parameters['indexed_parametric_coordinates']
        self.regularization_coeff = self.parameters['regularization_coeff']
    
    def compute(self):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model, lsdo_modules.ModuleCSDL}
            The csdl model or module that computes the model/operation outputs.
        '''
        associated_coords = {}
        index = 0
        for item in self.indexed_mesh:
            key = item[0]
            value = item[1]
            if key not in associated_coords.keys():
                associated_coords[key] = [[index], value]
            else:
                associated_coords[key][0].append(index)
                associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
            index += 1

        output_shape = (len(self.indexed_mesh), self.function.coefficients[self.indexed_mesh[0][0]].shape[-1])
        csdl_model = ModuleCSDL()
        function_values = csdl_model.register_module_input('function_values', shape=self.arguments['function_values'].shape)
        function_values = csdl.reshape(function_values, output_shape)
        csdl_model.register_module_output('test_function_values', function_values)
        for key, value in associated_coords.items(): # in the future, use submodels from the function spaces?
            if hasattr(self.function.space.spaces[key], 'compute_fitting_map'):
                fitting_matrix = self.function.space.spaces[key].compute_fitting_map(value[1])
            else:
                evaluation_matrix = self.function.space.spaces[key].compute_evaluation_map(value[1])
                if sps.issparse(evaluation_matrix):
                    evaluation_matrix = evaluation_matrix.toarray()
                if self.regularization_coeff is not None:
                    fitting_matrix = np.linalg.inv(evaluation_matrix.T@evaluation_matrix + self.regularization_coeff*np.eye(evaluation_matrix.shape[1]))@evaluation_matrix.T # tested with 1e-3
                else:
                    fitting_matrix = linalg.pinv(evaluation_matrix)
            fitting_matrix_csdl = csdl_model.register_module_input('fitting_matrix_'+key, val=fitting_matrix, shape = fitting_matrix.shape, computed_upstream=False)
            associated_function_values = csdl_model.create_output(name = key + '_fn_values', shape=(len(value[0]), output_shape[-1]))
            for i in range(len(value[0])):
                associated_function_values[i,:] = function_values[value[0][i], :]
            coefficients = csdl.matmat(fitting_matrix_csdl, associated_function_values)
            coeff_name = self.function.coefficients[key].name
            csdl_model.register_module_output(name = coeff_name, var = coefficients)
        
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

    def evaluate(self, function_values:Variable):
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
        self.name = f'{self.function.name}_inverse_evaluation'

        # Define operation arguments
        self.arguments = {'function_values':function_values}

        # Create the M3L variables that are being output

        return 


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

    def register_output(self, output:Variable, design_condition=None):
        '''
        Registers a state to the model group so the model group will compute and output this variable.
        If inverse_evaluate is called on a variable that already has a value, the residual is identified
        and an implicit solver is used.

        Parameters
        ----------
        output : Variable
            The variable that the model will output.
        '''
        if design_condition:
            prepend = design_condition.parameters['name']
        else:
            prepend = ''

        if isinstance(output, dict):
            for key, value in output.items():
                name = f'{prepend}{value.name}'
                self.outputs[name] = value
        elif  isinstance(output, list):
            for out in output:
                name = f'{prepend}{out.name}'
                self.outputs[name] = out
        elif type(output) is Variable:
            name = f'{prepend}{output.name}'
            self.outputs[name] = output
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
        """
        Adds all operations to the `self.operations` dictionary under their own name.
        """
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

    
    def assemble_modal(self) -> ModuleCSDL:
            # Assemble output states'output_jacobian_name'
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        model_csdl = ModuleCSDL()
        output_jacobian_names = []
        output_jacobian_vars = []

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

            if issubclass(type(operation), ImplicitOperation):
                # TODO: also take input_jacobian
                jacobian_csdl_model = operation.compute_derivatives()
                if type(jacobian_csdl_model) is csdl.Model:
                    model_csdl.add(submodel=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
                elif issubclass(type(jacobian_csdl_model), ModuleCSDL):
                    model_csdl.add_module(submodule=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None and input is not None:
                        model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting
                for key, value in operation.residual_partials.items():
                    model_csdl.add(submodel=Eig(size=operation.size), name=operation.name + '_' + key + '_eig', promotes=[])
                    
                    model_csdl.connect(operation_name + '.' + key, operation.name + '_' + key + '_eig' + '.A')
        self.modal_csdl_model = model_csdl
        return self.modal_csdl_model

class ImplicitAeroelasticModel(Model):
    def __init__(self, ) -> None:
        super().__init__()

        # initialize CADDEE CSDL model in which the fluid and solid are to be linked
        self.csdl_model = ModuleCSDL()

    def gather_operations_implicit(self, variable:Variable):
        if variable.operation is not None:
            operation = variable.operation
            if operation.name not in self.operations:
                self.operations[operation.name] = operation
                for input_name, input in operation.arguments.items():
                    self.gather_operations_implicit(input)

    def assemble(self):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations_implicit(output)
        
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

    def assemble_dynamic(self, initial_conditions:list, num_times:int, h_stepsize:float, parameters:list=None, integrator:str='RK4'):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations_implicit(output)
        # ODESystemModel = ModuleCSDL()
        # ODESystemModel.parameters.declare('num_nodes')
        # n = ODESystemModel.parameters['num_nodes']
        n = num_times
        residual_names = []
        residual_states = []
        # not collecting parameter names for now, could change this at some point

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ImplicitOperation):
                residual_names.append(operation.residual_name)
                residual_states.append(operation.residual_state)

        from ozone.api import ODEProblem

        ode_prob = ODEProblem(integrator, 'time-marching', num_times)

        if parameters is not None:
            for parameter in parameters:
                if parameter[1]:
                    ode_prob.add_parameter(parameter[0], dynamic=parameter[1], shape=num_times)
                else:
                    ode_prob.add_parameter(parameter[0])
        for i in range(len(residual_states)):
            ode_prob.add_state(residual_states[i], 
                                residual_names[i], 
                                initial_condition_name=residual_states[i]+'_0', 
                                output=residual_states[i]+'_integrated')
        ode_prob.add_times(step_vector='h')
        ode_prob.set_ode_system(AssembledODEModel)
                
        RunModel = csdl.Model()

        for ic in initial_conditions:
            RunModel.create_input(ic[0], ic[1])
        if parameters is not None:
            for parameter in parameters:
                RunModel.create_input(parameter[0], parameter[2])
        h_vec = np.ones(num_times-1)*h_stepsize
        RunModel.create_input('h', h_vec)
        RunModel.add(ode_prob.create_solver_model(ODE_parameters={'operations':self.operations}), 'prob')

        return RunModel

class AssembledODEModel(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('operations')
    def define(self):
        num_nodes = self.parameters['num_nodes']
        operations = self.parameters['operations']

        for operation_name, operation in operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute(num_nodes=num_nodes)

                if type(operation_csdl) is csdl.Model:
                    self.add(submodel=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
                elif issubclass(type(operation_csdl), ModuleCSDL):
                    self.add_module(submodule=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None:
                        self.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting

            if issubclass(type(operation), ImplicitOperation):
                # TODO: also take input_jacobian
                operation_csdl = operation.compute_residual(num_nodes=num_nodes)
                # promote these for connections in ozone - may need to make residual stuff a list
                promotions = operation.ode_parameters
                for i in range(len(operation.residual_names)):
                    promotions += [operation.residual_names[i][0], operation.residual_names[i][1]]
                # if issubclass(type(operation_csdl), csdl.Model):
                #     self.add(submodel=operation_csdl, name=operation_name, promotes=promotions)
                # elif issubclass(type(operation_csdl), ModuleCSDL):
                #     self.add_module(submodule=operation_csdl, name=operation_name, promotes=promotions)
                if issubclass(type(operation_csdl), csdl.Model):
                    self.add(submodel=operation_csdl, name=operation_name)
                elif issubclass(type(operation_csdl), ModuleCSDL):
                    self.add_module(submodule=operation_csdl, name=operation_name)
                else:
                    raise Exception(f"{operation.name}'s compute_residual() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None and input is not None:
                        self.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting


# This is a bit of a hack to get a caddee static model to do a modal assemble
class StructuralModalModel(Model):
    def assemble(self):
        # Assemble output states'output_jacobian_name'
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        model_csdl = ModuleCSDL()
        output_jacobian_names = []
        output_jacobian_vars = []

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

            if issubclass(type(operation), ImplicitOperation):
                # TODO: also take input_jacobian
                jacobian_csdl_model = operation.compute_derivatives()
                if type(jacobian_csdl_model) is csdl.Model:
                    model_csdl.add(submodel=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
                elif issubclass(type(jacobian_csdl_model), ModuleCSDL):
                    model_csdl.add_module(submodule=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None and input is not None:
                        model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting
                for key, value in operation.residual_partials.items():
                    model_csdl.add(submodel=Eig(size=operation.size), name=operation.name + '_' + key + '_eig', promotes=[])
                    
                    model_csdl.connect(operation_name + '.' + key, operation.name + '_' + key + '_eig' + '.A')
        self.csdl_model = model_csdl
        return self.csdl_model
