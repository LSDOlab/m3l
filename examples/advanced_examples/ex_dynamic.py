import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem
import csdl
import python_csdl_backend
import numpy as np
import m3l

# Pure ozone solution

class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)

        # Paramters are now inputs
        a = self.declare_variable('a', shape=(n))
        b = self.declare_variable('b', shape=(n))
        g = self.declare_variable('g', shape=(n))
        d = self.declare_variable('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - csdl.expand(d, n)*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('a', dynamic=True, shape=(self.num_times))
        self.add_parameter('b', dynamic=True, shape=(self.num_times))
        self.add_parameter('g', dynamic=True, shape=(self.num_times))
        # If dynamic != True, it is a static parameter. i.e, the parameter used in the ODE is constant through time.
        # Therefore, the shape does not depend on the number of timesteps
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0', output='x_integrated')
        self.add_times(step_vector='h')

        # Define ODE
        self.set_ode_system(ODESystemModel)

# The CSDL Model containing the ODE integrator
class RunModel(csdl.Model):
    def define(self):
        num_times = 401

        h_stepsize = 0.15

        # Initial condition for state
        y_0 = self.create_input('y_0', 2.0)
        x_0 = self.create_input('x_0', 2.0)

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        d = 0.5  # static parameter
        for t in range(num_times):
            a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
            b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
            g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

        # Add to csdl model which are fed into ODE Model
        ai = self.create_input('a', a)
        bi = self.create_input('b', b)
        gi = self.create_input('g', g)
        di = self.create_input('d', d)

        # Timestep vector
        h_vec = np.ones(num_times-1)*h_stepsize
        h = self.create_input('h', h_vec)

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times)

        self.add(ODEProblem.create_solver_model(), 'subgroup')

def run_ozone():
    # Simulator Object:
    sim = python_csdl_backend.Simulator(RunModel(), mode='rev')

    sim.run()

    # Plot
    plt.plot(sim['y_integrated'])
    plt.plot(sim['x_integrated'])
    plt.show()



# m3l solution:
# idk if both x and y need to be inputs to evaluate

class PredatorCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name'] + '_'
        n = self.parameters['num_nodes']
        y = self.create_input(name+'y', shape=n)
        x = self.create_input('x', shape=n)

        a = self.declare_variable(name+'a', shape=(n))
        b = self.declare_variable(name+'b', shape=(n))

        dy_dt = a*y - b*y*x
        self.register_output(name+'dy_dt', dy_dt)

class PreyCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name'] + '_'
        n = self.parameters['num_nodes']
        y = self.create_input('y', shape=n)
        x = self.create_input(name+'x', shape=n)

        g = self.declare_variable(name+'g', shape=(n,))
        d = self.declare_variable(name+'d')

        dx_dt = g*x*y - csdl.expand(d, n)*x
        self.register_output(name+'dx_dt', dx_dt)

class Predator(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='predator')
    def assign_atributes(self):
        self.name = self.parameters['name']
    def evaluate(self, x:m3l.Variable, y:m3l.Variable):
        self.assign_atributes()
        name = self.name + '_'
        self.ode_parameters = [name+'a', name+'b']
        self.arguments = {}
        self.inputs = {}
        self.arguments['x'] = x
        self.inputs['x'] = x
        self.residual_names = [(name+'y',name+'dy_dt',(1,))]
        residual = m3l.Variable(name='dy_dt', shape=(1,), operation=self) # think about not asking for duplicate info (residual names)
        return residual
    def compute_residual(self, num_nodes):
        csdl_model = PredatorCSDL(num_nodes=num_nodes, name=self.name)
        return csdl_model

class Prey(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='prey')
    def assign_atributes(self):
        self.name = self.parameters['name']
    def evaluate(self, x:m3l.Variable, y:m3l.Variable):
        self.assign_atributes()
        name = self.name+'_'
        self.ode_parameters = [name+'g', name+'d']
        self.arguments = {}
        self.inputs = {}
        self.arguments['y'] = y
        self.inputs['y'] = y
        self.residual_names = [(name+'x',name+'dx_dt',(1,))]
        residual = m3l.Variable(name='dx_dt', shape=(1,), operation=self)
        return residual
    def compute_residual(self, num_nodes):
        csdl_model = PreyCSDL(num_nodes=num_nodes, name=self.name)
        return csdl_model

def run_m3l():
    m3l_model = m3l.Model()

    predator = Predator()
    prey = Prey()

    y = m3l.Variable(name='predator_y', shape=(1,), operation=predator)
    x = m3l.Variable(name='prey_x', shape=(1,), operation=prey)

    predator_residual = predator.evaluate(x, y)
    prey_residual = prey.evaluate(x,y)

    # idk if I need to do both
    m3l_model.register_output(output=predator_residual)
    m3l_model.register_output(output=prey_residual)

    initial_conditions = [('predator_y_0', 2.0),('prey_x_0', 2.0)]
    num_times = 401
    h_stepsize = 0.15
    a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    d = .5  # static parameter
    for t in range(num_times):
        a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
        b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
        g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

    parameters = [('predator_a', True, a),('predator_b', True, b),('prey_g', True, g), ('prey_d', False, d)]


    dynamic_model = m3l_model.assemble_dynamic(initial_conditions=initial_conditions,
                                            num_times=num_times,
                                            h_stepsize=h_stepsize,
                                            parameters=parameters
                                            )
    sim = python_csdl_backend.Simulator(dynamic_model, analytics=True)
    sim.run()
    # Plot
    plt.plot(sim['predator_y_integrated'])
    plt.plot(sim['prey_x_integrated'])
    plt.show()

def run_m3l_v2():
    m3l_model = m3l.DynamicModel()

    predator = Predator()
    prey = Prey()

    y = m3l.Variable(name='predator_y', shape=(1,), operation=predator)
    x = m3l.Variable(name='prey_x', shape=(1,), operation=prey)

    predator_residual = predator.evaluate(x, y)
    prey_residual = prey.evaluate(x,y)

    # idk if I need to do both
    m3l_model.register_output(output=predator_residual)
    m3l_model.register_output(output=prey_residual)

    initial_conditions = [('predator_y_0', 2.0),('prey_x_0', 2.0)]
    num_times = 401
    h_stepsize = 0.15
    a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    d = .5  # static parameter
    for t in range(num_times):
        a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
        b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
        g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

    parameters = [('predator_a', True, a),('predator_b', True, b),('prey_g', True, g), ('prey_d', False, d)]

    m3l_model.set_dynamic_options(initial_conditions=initial_conditions,
                                  num_times=num_times,
                                  h_stepsize=h_stepsize,
                                  parameters=parameters)
    dynamic_model = m3l_model.assemble()
    sim = python_csdl_backend.Simulator(dynamic_model, analytics=True)
    sim.run()
    # Plot
    plt.plot(sim['predator_y_integrated'])
    plt.plot(sim['prey_x_integrated'])
    plt.show()

# run_ozone()
# run_m3l()
run_m3l_v2()
