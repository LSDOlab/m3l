from dataclasses import dataclass, is_dataclass, asdict, field
from typing import Any, List
import gc
import numpy as np
import scipy.sparse as sps
from scipy import linalg
# import array_mapper as am
# import scipy.sparse as sps
import csdl
from typing import Union
from m3l.utils.base_class import OperationBase
from ozone.api import ODEProblem

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


class Operation(OperationBase):
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
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('nested_connection_name', types=bool, default=False)


    def assign_attributes(self):
        '''
        Assigns class attributes to make class more like standard python class.
        '''
        self.m3l_inputs = []
        self.name = self.parameters['name']
        # self.nested_connection_name = self.parameters['nested_connection_name']
        pass
    
    def compute(self):
        '''
        Creates the CSDL model to compute the model output.

        Returns
        -------
        csdl_model : {csdl.Model}
            The csdl model that computes the model/operation outputs.
        '''
        pass

    def compute_derivates(self):
        '''
        -- optional --
        Creates the CSDL model to compute the derivatives of the model outputs. This is only needed for dynamic analysis.
        For now, I would recommend coming back to this.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
        csdl_model : {csdl.Model}
            The csdl model  that computes the residuals.
        '''
        pass

    # only needed for dynamic stuff
    def compute_derivatives(self):
        '''
        Solver developer API method for the backend to call.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
        invariant_matrix_csdl_model : {csdl.Model}
            The csdl model  that computes the invariant matrix for the SIFR methodology.
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
    variable_counter = 0
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
    shape : tuple
    name : str = None
    operation : Operation = None
    value : np.ndarray = None
    dv_flag : bool = False
    lower : Union[int, float, np.ndarray, None] = None
    upper : Union[int, float, np.ndarray, None] = None
    scaler : Union[int, float, None] = None
    equals : Union[int, float, np.ndarray, None] = None
    description : str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f'{Variable.variable_counter}'
        Variable.variable_counter += 1

    def __getitem__(self, indices):
        import m3l
        return m3l.variable_get_item(self, indices)

    def __setitem__(self, indices, value):
        import m3l
        new_me = m3l.variable_set_item(self, indices, value)

        self.name = new_me.name
        self.operation = new_me.operation
        self.value = new_me.value

    
    def __len__(self):
        # if self.operation is None:
            # print(self.name, self.shape)
        # else:
            # print(self.operation.name, self.name, self.shape)


        return self.shape[0]

    def __add__(self, other):
        import m3l
        return m3l.add(self, other)
    
    def __radd__(self, other):
            return self.__add__(other=other)
    
    def __sub__(self, other):
        import m3l
        return m3l.subtract(self, other)
    
    def __rsub__(self, other):
        return -1*self.__sub__(other=other)
    
    def __mul__(self, other):
        import m3l
        return m3l.multiply(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other=other)
    
    def __truediv__(self, other):
        import m3l
        return m3l.divide(self, other)
    
    def __str__(self):
        return str(self.value)
    
    def __pow__(self, other):
        import m3l
        return m3l.power(self, other)
    
    def reshape(self, shape:tuple):
        '''
        Reshapes the variable.

        Parameters
        ----------
        shape : tuple
            The new shape of the variable.
        '''
        import m3l
        return m3l.reshape(self, shape)
    
    def copy(self):
        '''
        Returns a copy of the variable.
        '''
        import m3l
        return m3l.copy(self)


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

    def __call__(self, mesh : Variable) -> Variable:
        return self.evaluate(mesh)

    def evaluate(self, mesh : Variable) -> Variable:
        '''
        Evaluate the function at a given set of nodal locations.

        Parameters
        ----------
        mesh : Variable
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

        # csdl_map = csdl.Model()
        # function_coefficients = csdl_map.declare_variable(self.coefficients.name, shape=(num_coefficients, self.coefficients.shape[-1]))
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

        csdl_map = csdl.Model()
        function_values_csdl = csdl_map.declare_variable(function_values.name, shape=(num_values,3))
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
        self.parameters.declare('mesh', types=Variable)

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
        csdl_model : {csdl.Model}
            The csdl model  that computes the model/operation outputs.
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

        csdl_map = csdl.Model()
        function_coefficients = csdl_map.declare_variable(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
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
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
    coefficients : dict[str, Variable] = None

    def __call__(self, mesh : Variable) -> Variable:
        return self.evaluate(mesh)

    def evaluate(self, indexed_parametric_coordinates) -> Variable:
        '''
        Evaluate the function at a given set of nodal locations.

        Parameters
        ----------
        mesh : Variable
            The mesh to evaluate over.

        Returns
        -------
        function_values : FunctionValues
            A variable representing the evaluated function values.
        '''
        function_evaluation_model = IndexedFunctionEvaluation(function=self, indexed_parametric_coordinates=indexed_parametric_coordinates)
        function_values = function_evaluation_model.evaluate()
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
    
    def compute(self, num_nodes:int=1):
        '''
        Creates the CSDL model to compute the function evaluation.

        Returns
        -------
        csdl_model : {csdl.Model}
            The csdl model  that computes the model/operation outputs.
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
        csdl_map = csdl.Model()
        points = csdl_map.create_output(output_name, shape=output_shape)
        
        coefficients_csdl = {} 
        for key, coefficients in self.function.coefficients.items():
            num_coefficients = np.prod(coefficients.shape[:-1])
            if coefficients.value is None:
                coefficients_csdl[key] = csdl_map.declare_variable(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]))
            else:
                coefficients_csdl[key] = csdl_map.declare_variable(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
                                                                    val=coefficients.value.reshape((-1, coefficients.shape[-1])))

        for key, value in associated_coords.items():
            evaluation_matrix = self.function.space.spaces[key].compute_evaluation_map(value[1])
            if sps.issparse(evaluation_matrix):
                evaluation_matrix = evaluation_matrix.toarray()
            evaluation_matrix_csdl = csdl_map.declare_variable('evaluation_matrix_'+key, val=evaluation_matrix, shape = evaluation_matrix.shape, computed_upstream=False)
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
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
        csdl_model : {csdl.Model}
            The csdl model  that computes the model/operation outputs.
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
        csdl_map = csdl.Model()
        points = csdl_map.create_output(output_name, shape=output_shape)
        
        coefficients_csdl = {} 
        for key, coefficients in self.function.coefficients.items():
            num_coefficients = np.prod(coefficients.shape[:-1])
            if coefficients.value is None:
                coefficients_csdl[key] = csdl_map.declare_variable(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]))
            else:
                coefficients_csdl[key] = csdl_map.declare_variable(coefficients.name, shape=(num_coefficients, coefficients.shape[-1]),
                                                                    val=coefficients.value.reshape((-1, coefficients.shape[-1])))

        for key, value in associated_coords.items():
            evaluation_matrix_u = self.function.space.spaces[key].compute_evaluation_map(value[1], parametric_derivative_order=(1,0))
            evaluation_matrix_v = self.function.space.spaces[key].compute_evaluation_map(value[1], parametric_derivative_order=(0,1))
            if sps.issparse(evaluation_matrix_u):
                evaluation_matrix_u = evaluation_matrix_u.toarray()
                evaluation_matrix_v = evaluation_matrix_v.toarray()

            evaluation_matrix_u_csdl = csdl_map.declare_variable('evaluation_matrix_u_'+key, val=evaluation_matrix_u, shape = evaluation_matrix_u.shape, computed_upstream=False)
            evaluation_matrix_v_csdl = csdl_map.declare_variable('evaluation_matrix_v_'+key, val=evaluation_matrix_v, shape = evaluation_matrix_v.shape, computed_upstream=False)

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
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
        csdl_model : {csdl.Model}
            The csdl model that computes the model/operation outputs.
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
        csdl_model = csdl.Model()
        function_values = csdl_model.declare_variable('function_values', shape=self.arguments['function_values'].shape)
        function_values = csdl.reshape(function_values, output_shape)
        csdl_model.register_output('test_function_values', function_values)
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
            fitting_matrix_csdl = csdl_model.declare_variable('fitting_matrix_'+key, val=fitting_matrix, shape = fitting_matrix.shape, computed_upstream=False)
            associated_function_values = csdl_model.create_output(name = key + '_fn_values', shape=(len(value[0]), output_shape[-1]))
            for i in range(len(value[0])):
                associated_function_values[i,:] = function_values[value[0][i], :]
            coefficients = csdl.matmat(fitting_matrix_csdl, associated_function_values)
            coeff_name = self.function.coefficients[key].name
            csdl_model.register_output(name = coeff_name, var = coefficients)
        
        return csdl_model

    def compute_derivates(self):
        '''
        -- optional --
        Creates the CSDL model to compute the derivatives of the model outputs. This is only needed for dynamic analysis.
        For now, I would recommend coming back to this.

        Returns
        -------
        derivatives_csdl_model : {csdl.Model}
            The csdl model  that computes the derivatives of the model/operation outputs.
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
        # self.operations = []
        self.outputs = {}
        self.parameters = None
        self.constraints = []
        self.objective = None
        self.user_inputs = []

    # def add(self, submodel:Model, name:str):
    #     self.models[name] = submodel

    def create_input(self, name: str, val: Union[int, float, np.ndarray], shape: Union[tuple, None] = None,
                     prefix: str = '', dv_flag: bool = False,
                     upper: Union[int, float, np.ndarray, None] = None,
                     lower: Union[int, float, np.ndarray, None] = None,
                     scaler: Union[int, float] = None):
        """
        Method to create M3L variables and specify design variables.

        Parameters:
        ----------
        name : str
            Name of the variable.
        
        val : int, float, or np.ndarray
            Value of the m3l variable.
        
        shape : tuple, None optional (default: None)
            Shape of the variable specified as a tuple.
        
        prefix : str, optional
            Optional variable prefix. Recommended to create a unique namespace for variables of the same kind.
        
        dv_flag : bool, optional, default: False
            Specify whether a certain variable is a design variable for optimization.
        
        upper : int, float, np.ndarray, or None, optional, default: None
            Set an upper bound on a design variable.
        
        lower : int, float, np.ndarray, or None, optional, default: None
            Set a lower bound on a design variable.
        
        scaler : int or float, optional
            Scale design variables.
        
        Returns:
        -------
        m3l.Variable
            An instance of an M3L Variable.
        """

        # operation_name = self.name


        # for var in self.m3l_inputs:
        #     existing_name = var.name
        #     if f"{operation_name}_{name}" == existing_name:
        #         raise ValueError(f"Variable '{name}' already exists for operation '{operation_name}'. Please use a uniqe name.") 

        for user_input in self.user_inputs:
            if name == user_input.name:
                raise ValueError(f"Variable '{name}' alredy exists as an input. Please make sure to give each m3l input variable a unique name.")
        
        if not isinstance(val, (int, float, np.ndarray)):
            raise TypeError('Invalid type for value. Must be int, float or np.ndarray')

        if shape:
            var_shape = shape
            if isinstance(val, np.ndarray):
                if var_shape != val.shape:
                    error_message = f"Shape mismatch: variable '{name}' has shape {val.shape} but the specified shape is {var_shape}. If you would like to expand the array, consider using 'np.newaxis' in combination with 'np.repeat'. See the example below \n\n"
                    
                    error_message += '''\
                        a = np.array([1, 2, 3])
                        a_exp = a[np.newaxis, :].repeat(n, axis=0)

                        Here, a_exp will have dimensions (n, 3).
                    '''
                    raise ValueError(error_message)

        else:
            if isinstance(val, (int, float)):
                var_shape = (1, )
            elif isinstance(val, np.ndarray):
                var_shape = val.shape
            else:
                raise NotImplementedError

        m3l_var = Variable(
            # name=f"{operation_name}_{name}",
            name=name,
            value=val,
            shape=var_shape,
            operation=None,
            dv_flag=dv_flag,
            upper=upper, 
            lower=lower,
            scaler=scaler,
        )

        # self.m3l_inputs.append(m3l_var)
        self.user_inputs.append(m3l_var)

        return m3l_var


    def register_output(self, output:Variable, string_name : Union[str, None]=None):
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
            if string_name:
                for key, value in output.items():
                    if value.operation:
                        name = f'{string_name}_{value.operation.name}_{value.name}'
                        self.outputs[name] = value
                    else:
                        name = f'{string_name}_{value.name}'
                        self.outputs[name] = value
                        raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
            else:
                for key, value in output.items():
                    if value.operation:
                        name = f'{value.operation.name}_{value.name}'
                        self.outputs[name] = value
                    else:
                        name = f'{value.name}'
                        self.outputs[name] = value
                        raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
      
        elif  isinstance(output, list):
            if string_name:
                for out in output:
                    if out.operation:
                        name = f'{string_name}_{out.operation.name}_{out.name}'
                        self.outputs[name] = out
                    else:
                        name = f'{string_name}_{out.name}'
                        self.outputs[name] = out
                        raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
            else:
                for out in output:
                    if out.operation:
                        name = f'{out.operation.name}_{out.name}'
                        self.outputs[name] = out
                    else:
                        name = f'{out.name}'
                        self.outputs[name] = out
                        raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
        
        elif type(output) is Variable:
            if string_name:
                if output.operation:
                    name = f'{string_name}_{output.operation.name}_{output.name}'
                    self.outputs[name] = output
                else:
                    name = f'{string_name}_{output.name}'
                    self.outputs[name] = output
                    raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")

            else:
                if output.operation:
                    # print(output.name)
                    # print(output.operation.name)
                    name = f'{output.operation.name}_{output.name}'
                    self.outputs[name] = output
                else:
                    name = f'{output.name}'
                    self.outputs[name] = output
                    raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
                
       
        elif is_dataclass(output):
            # attributes = asdict(output)
            attributes = output.__dict__
            if string_name:
                for key, value in attributes.items():
                    if is_dataclass(value):
                        self.register_output(value, string_name=string_name)
                    
                    elif isinstance(value, dict):
                        self.register_output(value, string_name=string_name)

                    elif isinstance(value, list):
                        self.register_output(value, string_name=string_name)

                    elif not isinstance(value, Variable):
                        pass

                    else:
                        if value.operation:
                            name = f'{string_name}_{value.operation.name}_{key}'
                            self.outputs[name] = value
                        else:
                            name = f'{string_name}_{key}'
                            self.outputs[name] = value
                            raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
            else:
                for key, value in attributes.items():
                    if is_dataclass(value):
                        self.register_output(value)

                    elif isinstance(value, dict):
                        self.register_output(value)

                    elif isinstance(value, list):
                        self.register_output(value)

                    elif not isinstance(value, Variable):
                        pass
                    else:
                        if value.operation:
                            name = f'{value.operation.name}_{key}'
                            self.outputs[name] = value
                        else:
                            name = f'{key}'
                            self.outputs[name] = value
                            raise Warning(f"Variable {value.name} is not computed from any upstream operation. Registering this as an output doesn't accomplish anything.")
        else:
            print(type(output))
            raise NotImplementedError


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

    def add_constraint(self, m3l_var: Variable, lower=None, upper=None, equals=None, scaler=None):
        """
        Method to add constraints based on high-level m3l variables
        """
        m3l_var.equals = equals
        m3l_var.lower = lower 
        m3l_var.upper = upper
        m3l_var.scaler = scaler

        self.constraints.append(m3l_var)

    def add_objective(self, m3l_var: Variable, scaler=None):
        """
        Method to define an objective
        """
        m3l_var.scaler = scaler
        self.objective = m3l_var

    # def gather_operations(self, variable:Variable):
    #     # print(self.depth)
    #     self.depth += 1
    #     if variable:
    #         # print(variable)
    #         if variable.operation is not None:
    #             operation = variable.operation
    #             for input_name, input in operation.arguments.items():
    #                 if input is not None:
    #                     self.gather_operations(input)

    #             if operation.name not in self.operations:
    #                 self.operations[operation.name] = operation
    #         else:
    #             pass
    #             print(f'Variable {variable.name} is not part of an operation')
    
    # def gather_operations(self, variable:Variable):
    #     print(self.depth)
    #     self.depth += 1
    #     if variable:
    #         # print(variable)
    #         if variable.operation is not None:
    #             operation = variable.operation
    #             for input_name, input in operation.arguments.items():
    #                 if input is not None:
    #                     self.gather_operations(input)

    #             is_already_added = self.check_if_operation_has_been_added(operation)

    #             if not is_already_added:
    #                 self.operations.append(operation)
    #         else:
    #             pass

    # def gather_operations(self, variable:Variable):
    #     '''
    #     Non-recursive implementation of gather operations to avoid recursion depth limit.
    #     '''
    #     all_none = False
    #     while not all_none:

    #     operation = variable.operation
    #     while operation is not None:
    #         for input_name, input in operation.arguments.items():
    #             if input.operation is None:
    #                 continue
    #             operation = input.operation
    #             operation_is_already_added = self.check_if_operation_has_been_added(operation)
    #             if not operation_is_already_added:
    #                 self.operations.append(operation)

    #     self.operations.reverse()   # Do I want to do this? I think so. It's depth-based either way which is not that intuitive but easier.

    def check_if_operation_has_been_added(self, operation:Operation):
        '''
        Checks if this operation has already been added to the model.
        '''
        for model_operation in self.operations:
            if operation is model_operation:
                return True
            
    def check_if_variable_is_in_list(self, variable:Variable, stack:list):
        '''
        Checks if this operation has already been added to the model.
        '''
        for variable_in_stack in stack:
            if variable is variable_in_stack:
                return True

    # def gather_operations(self, variable: Variable):
    #     total_stack = []
    #     stack = []
    #     stack.append(variable)

    #     while stack:
    #         current_variable = stack.pop()
    #         if current_variable:
    #             if current_variable.operation is not None:
    #                 operation = current_variable.operation

    #                 for input_name, input in operation.arguments.items():
    #                     if input is not None:
    #                         variable_is_added_to_stack = self.check_if_variable_is_in_list(input, total_stack)

    #                         if not variable_is_added_to_stack:
    #                             stack.append(input)
    #                             total_stack.append(input)

    #                 operation_is_added = self.check_if_operation_has_been_added(operation)

    #                 if not operation_is_added:
    #                     # self.operations[operation.name] = operation
    #                     self.operations.append(operation)
    #             else:
    #                 pass

    #     self.operations.reverse()

    def gather_operations(self, variable: Variable):
        from collections import deque
        total_stack = []
        stack = deque([variable])

        while stack:
            current_variable = stack.popleft()

            if current_variable:
                if current_variable.operation is not None:
                    operation = current_variable.operation
                    is_already_added = self.check_if_operation_has_been_added(operation)

                    # if not is_already_added:
                    if operation.name not in self.operations:
                        self.operations[operation.name] = operation

                    for input_name, input in operation.arguments.items():
                        variable_is_already_added = self.check_if_variable_is_in_list(input, total_stack)
                        if not variable_is_already_added:
                            stack.append(input)
                            total_stack.append(input)

        # self.operations.reverse()


    def gather_operations_implicit(self, variable:Variable):
        if variable.operation is not None:
            operation = variable.operation
            if operation.name not in self.operations:
                self.operations[operation.name] = operation
                for input_name, input in operation.arguments.items():
                    self.gather_operations_implicit(input)

    # def assemble(self):
    #     # Assemble output states
    #     for output_name, output in self.outputs.items():
    #         self.gather_operations(output)
        
    #     model_csdl = csdl.Model()

    #     for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process

    #         if type(operation.operation_csdl) is csdl.Model:
    #             model_csdl.add(submodel=operation.operation_csdl, name=operation.name, promotes=[]) # should I suppress promotions here?
    #         else: # type(operation.operation_csdl) is csdl.Model:
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
    

    def assemble(self) -> csdl.Model:
        self.depth = 0
        
        # print(self.outputs.items())
        # exit()
        # Assemble output states
        for output_name, output in self.outputs.items():
            # print('------------------------------------------', output_name)
            self.gather_operations(output)
        
        model_csdl = csdl.Model()

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
        # for operation in self.operations:
            operation_name = operation.name
            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute()
                if issubclass(type(operation_csdl), csdl.Model):
                    # print(operation_name)
                    model_csdl.add(submodel=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here-Yes?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type : {type(operation_csdl)}.")


                if not operation.arguments and 'connect_from' in operation.parameters:
                    for i in range(len(operation.parameters['connect_from'])):
                        connect_from = operation.parameters['connect_from'][i]
                        connect_to = operation.parameters['connect_to'][i]
                        model_csdl.connect(connect_from, connect_to) 

                else:
                    for input_name, input in operation.arguments.items():
                        if input:
                            if input.operation is not None: # If the input is associated with an operation
                                    model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name)    
                            else: # if there is no input associated with an operation (i.e., top-level, user-defined inputs)
                                if input not in self.user_inputs:
                                    model_csdl.create_input(input.name, val=input.value)

                                model_csdl.connect(input.name, operation_name+"."+input_name) 
                    
                                       
            if issubclass(type(operation), ImplicitOperation):
                # TODO: also take input_jacobian
                jacobian_csdl_model = operation.compute_derivatives()
                if issubclass(type(jacobian_csdl_model), csdl.Model):
                # if type(jacobian_csdl_model) is csdl.Model:
                    model_csdl.add(submodel=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None and input is not None:
                        model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting
                for key, value in operation.residual_partials.items():
                    model_csdl.add(submodel=Eig(size=operation.size), name=operation.name + '_' + key + '_eig', promotes=[])
                    
                    model_csdl.connect(operation_name + '.' + key, operation.name + '_' + key + '_eig' + '.A')
        # Create any user-defined inputs
        for input in self.user_inputs:
            var_name = input.name
            var_val = input.value
            var_shape = input.shape
            dv_flag = input.dv_flag

            model_csdl.create_input(name=var_name, val=var_val, shape=var_shape)

            if dv_flag:
                lower = input.lower
                upper = input.upper
                scaler = input.scaler
                model_csdl.add_design_variable(var_name, lower=lower, upper=upper, scaler=scaler)

        
        # Add constraints and objective
        for var in self.constraints:
            var_name = var.name
            lower = var.lower
            upper = var.upper
            equals = var.equals
            scaler = var.scaler
            operation = var.operation
            operation_name = operation.name

            model_csdl.add_constraint(name=f"{operation_name}.{var_name}", lower=lower, upper=upper, equals=equals, scaler=scaler)

        if self.objective:
            var_name = self.objective.name
            scaler = self.objective.scaler
            operation = self.objective.operation
            operation_name = self.objective.operation.name
            model_csdl.add_objective(name=f"{operation_name}.{var_name}", scaler=scaler)

        csdl_model = model_csdl
        # self.csdl_model = model_csdl

        # del self.operations
        # del self
        # gc.collect()
        return csdl_model


    def assemble_csdl(self) -> csdl.Model:
        csdl_model = self.assemble()
        # del self
        # gc.collect()
        return csdl_model #self.csdl_model
    

    def assemble_derivative_model(self) -> csdl.Model:
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        derivative_model_csdl = csdl.Model()

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ExplicitOperation):
                derivative_operation_csdl = operation.compute_derivatives()
                if derivative_operation_csdl is not None:
                    if issubclass(type(derivative_operation_csdl), csdl.Model):
                        derivative_model_csdl.add(submodel=derivative_operation_csdl, name=operation_name, promotes=[])
                    else:
                        raise Exception(f"{operation.name}'s compute_derivatives() method is returning an invalid model type : {type(derivative_operation_csdl)}.")
                else:
                    # Maybe if compute_derivatives is None, then assume it's linear?
                    raise Exception(f"{operation.name}'s compute_derivatives() method is returning None. Please make sure to return a valid model.")

                if not operation.arguments and 'connect_from' in operation.parameters:
                    for i in range(len(operation.parameters['connect_from'])):
                        connect_from = operation.parameters['connect_from'][i]
                        connect_to = operation.parameters['connect_to'][i]
                        derivative_model_csdl.connect(connect_from, connect_to) 

                else:
                    for input_name, input in operation.arguments.items():
                        if input:
                            if input.operation is not None: # If the input is associated with an operation
                                    derivative_model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name)    
                            else: # if there is no input associated with an operation (i.e., top-level, user-defined inputs)
                                if input not in self.user_inputs:
                                    derivative_model_csdl.create_input(input.name, val=input.value)

                                derivative_model_csdl.connect(input.name, operation_name+"."+input_name) 
                    
            # Just gonna ignore this for now
            # if issubclass(type(operation), ImplicitOperation):
            #     # TODO: also take input_jacobian
            #     jacobian_csdl_model = operation.compute_derivatives()
            #     if issubclass(type(jacobian_csdl_model), csdl.Model):
            #     # if type(jacobian_csdl_model) is csdl.Model:
            #         derivative_model_csdl.add(submodel=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
            #     elif issubclass(type(jacobian_csdl_model), ModuleCSDL):
            #         derivative_model_csdl.add_module(submodule=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
            #     else:
            #         raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

            #     for input_name, input in operation.arguments.items():
            #         if input.operation is not None and input is not None:
            #             derivative_model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting
            #     for key, value in operation.residual_partials.items():
            #         derivative_model_csdl.add(submodel=Eig(size=operation.size), name=operation.name + '_' + key + '_eig', promotes=[])
                    
            #         derivative_model_csdl.connect(operation_name + '.' + key, operation.name + '_' + key + '_eig' + '.A')

        # Create any user-defined inputs
        for input in self.user_inputs:
            var_name = input.name
            var_val = input.value
            var_shape = input.shape
            dv_flag = input.dv_flag

            derivative_model_csdl.create_input(name=var_name, val=var_val, shape=var_shape)

            if dv_flag:
                lower = input.lower
                upper = input.upper
                scaler = input.scaler
                derivative_model_csdl.add_design_variable(var_name, lower=lower, upper=upper, scaler=scaler)

        
        # Add constraints and objective
        for var in self.constraints:
            var_name = var.name
            lower = var.lower
            upper = var.upper
            equals = var.equals
            scaler = var.scaler
            operation = var.operation
            operation_name = operation.name

            derivative_model_csdl.add_constraint(name=f"{operation_name}.{var_name}", lower=lower, upper=upper, equals=equals, scaler=scaler)

        if self.objective:
            var_name = self.objective.name
            scaler = self.objective.scaler
            operation = self.objective.operation
            operation_name = self.objective.operation.name
            derivative_model_csdl.add_objective(name=f"{operation_name}.{var_name}", scaler=scaler)

        return derivative_model_csdl

    # parameters - list[(name, dynamic?, value(s))]
    def assemble_dynamic(self, initial_conditions:list, num_times:int, h_stepsize:float, parameters:list=None, integrator:str='RK4'):
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations_implicit(output)
        # ODESystemModel = csdl.Model()
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

    def assemble_modal(self) -> csdl.Model:
            # Assemble output states'output_jacobian_name'
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations(output)
        
        model_csdl = csdl.Model()
        output_jacobian_names = []
        output_jacobian_vars = []

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute()

                if type(operation_csdl) is csdl.Model:
                    model_csdl.add(submodel=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
                else:
                    raise Exception(f"{operation.name}'s compute() method is returning an invalid model type.")

                for input_name, input in operation.arguments.items():
                    if input.operation is not None:
                        model_csdl.connect(input.operation.name+"."+input.name, operation_name+"."+input_name) # when not promoting
                    else:
                        model_csdl.connect(input.name, operation_name+"."+input_name)

            if issubclass(type(operation), ImplicitOperation):
                # TODO: also take input_jacobian
                jacobian_csdl_model = operation.compute_derivatives()
                if type(jacobian_csdl_model) is csdl.Model:
                    model_csdl.add(submodel=jacobian_csdl_model, name=operation_name, promotes=[]) # should I suppress promotions here?
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

class DynamicModel(Model):
    def set_dynamic_options(self, 
                            initial_conditions:list, 
                            num_times:int, 
                            h_stepsize:float,
                            int_naming:tuple = ('','_integrated'),
                            parameters:list=None, 
                            integrator:str='RK4',
                            approach:str='time-marching checkpointing',
                            profile_outputs:list=None, 
                            profile_system=None, 
                            profile_parameters:dict=None,
                            copycat_profile:bool=False,
                            post_processor=None,
                            pp_vars:list=None):
        self.initial_conditions = initial_conditions
        self.num_times = num_times
        self.h_stepsize = h_stepsize
        self.ODE_parameters = parameters
        self.integrator = integrator
        self.profile_outputs = profile_outputs
        self.profile_system = profile_system
        self.profile_parameters = profile_parameters
        self.copycat_profile = copycat_profile
        self.approach = approach
        self.post_processor = post_processor
        self.pp_vars = pp_vars
        self.int_naming = int_naming
    def assemble(self, return_operation:bool=False):
        initial_conditions = self.initial_conditions
        num_times = self.num_times
        h_stepsize = self.h_stepsize
        parameters = self.ODE_parameters
        integrator = self.integrator
        int_naming = self.int_naming
        # Assemble output states
        for output_name, output in self.outputs.items():
            self.gather_operations_implicit(output)
        residual_names = []
        # not collecting parameter names for now, could change this at some point

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ImplicitOperation):
                residual_names += operation.residual_names

        ode_prob = ODEProblem(integrator, self.approach, num_times)

        if parameters is not None:  # do a register output and addemble model for each parameter that's a m3l var
            for parameter in parameters:
                if parameter[1]:
                    ode_prob.add_parameter(parameter[0], dynamic=parameter[1], shape=parameter[2].shape)
                else:
                    ode_prob.add_parameter(parameter[0])
        for i in range(len(residual_names)):
            ode_prob.add_state(residual_names[i][0], 
                                residual_names[i][1],
                                shape = residual_names[i][2],
                                initial_condition_name=residual_names[i][0]+'_0', 
                                output=int_naming[0] + residual_names[i][0] + int_naming[1])
                                # output=residual_names[i][0]+'_integrated')
        ode_prob.add_times(step_vector='h')
        ode_prob.set_ode_system(AssembledODEModel)
        # profile outputs
        if self.copycat_profile:
            self.profile_system = AssembledODEModel
            self.profile_parameters = {'operations':self.operations}
        if self.profile_outputs is not None:
            for profile_output in self.profile_outputs:
                ode_prob.add_profile_output(profile_output[0], shape=profile_output[1])
            ode_prob.set_profile_system(self.profile_system)
        
                
        RunModel = csdl.Model()

        for ic in initial_conditions:
            RunModel.create_input(ic[0], ic[1])
        if parameters is not None:
            parameter_model = Model()
            add_flag = False
            for parameter in parameters:
                if type(parameter[2]) is Variable:
                    add_flag = True
                    parameter_model.register_output(parameter[2])
            parameter_model_csdl = parameter_model.assemble()
            if add_flag:
                RunModel.add(parameter_model_csdl, name='input_model')
            for parameter in parameters:
                if not type(parameter[2]) is Variable:
                    RunModel.create_input(parameter[0], parameter[2])
                else:
                    RunModel.create_input(parameter[0], shape=parameter[2].shape)
                    in_name = 'input_model.' + parameter[2].operation.name + '.' + parameter[2].name
                    RunModel.connect(in_name, parameter[0])
        h_vec = np.ones(num_times-1)*h_stepsize
        RunModel.create_input('h', h_vec)
        if self.profile_parameters is not None:
            RunModel.add(ode_prob.create_solver_model(ODE_parameters={'operations':self.operations}, profile_parameters=self.profile_parameters), 'prob')
        else:
            RunModel.add(ode_prob.create_solver_model(ODE_parameters={'operations':self.operations}), 'prob')

        if self.post_processor is not None:
            RunModel.add(self.post_processor, name='post_processor')

        self.csdl_model=RunModel
        if return_operation:
            operation = DynamicOperation()
            operation.set_model(RunModel)
            outputs = []
            if self.pp_vars is not None:
                for val in self.pp_vars:
                    outputs.append(Variable(val[0], val[1], operation=operation))
            if self.profile_outputs is not None:
                for val in self.profile_outputs:
                    outputs.append(Variable(val[0], val[1], operation=operation))
            for val in residual_names:
                # outputs.append(Variable(val[0] + '_integrated', val[2], operation=operation))
                outputs.append(Variable('op_' + val[0], val[2], operation=operation))

            operation.set_outputs(outputs)
            return operation
        else:
            return RunModel

class DynamicOperation(ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', default='operation')
        self.name = self.parameters['name']
    def set_model(self, model):
        self.csdl_model = model
    def set_outputs(self, outputs):
        self.outputs = outputs
    def evaluate(self):
        self.arguments = {}
        return tuple(self.outputs)
    def compute(self):
        return self.csdl_model


class AssembledODEModel(csdl.Model):
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
                if issubclass(type(operation_csdl), csdl.Model):
                    self.add(submodel=operation_csdl, name=operation_name)
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
        
        model_csdl = csdl.Model()
        output_jacobian_names = []
        output_jacobian_vars = []

        for operation_name, operation in self.operations.items():   # Already in correct order due to recursion process
            if issubclass(type(operation), ExplicitOperation):
                operation_csdl = operation.compute()

                if type(operation_csdl) is csdl.Model:
                    model_csdl.add(submodel=operation_csdl, name=operation_name, promotes=[]) # should I suppress promotions here?
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