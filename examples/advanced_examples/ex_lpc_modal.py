import caddee.api as cd
from python_csdl_backend import Simulator
import numpy as np
import array_mapper as am
from aframe.core.beam_module import EBBeam, LinearBeamMesh, BeamM3LDisplacement
# from modopt.snopt_library import SNOPT
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import csdl
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
from caddee import GEOMETRY_FILES_FOLDER
import m3l
import lsdo_geo as lg
import aframe.core.beam_module as ebbeam

caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)

file_name = 'lift_plus_cruise_final.stp'

spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=GEOMETRY_FILES_FOLDER / file_name)
spatial_rep.refit_geometry(file_name=GEOMETRY_FILES_FOLDER / file_name)
# spatial_rep.plot(plot_types=['mesh'])





# wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing_1']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
# wing.plot()

# Adding components
sys_rep.add_component(wing)





# sys_param.add_geometry_parameterization(ffd_set)
sys_param.setup()



# wing mesh
num_wing_vlm = 11
num_chordwise_vlm = 5
point00 = np.array([12.356, 25.250, 7.618 + 0.1]) # * ft2m # Right tip leading edge
point01 = np.array([13.400, 25.250, 7.617 + 0.1]) # * ft2m # Right tip trailing edge
point10 = np.array([8.892,    0.000, 8.633 + 0.1]) # * ft2m # Center Leading Edge
point11 = np.array([14.332,   0.000, 8.439 + 0.1]) # * ft2m # Center Trailing edge
point20 = np.array([12.356, -25.250, 7.618 + 0.1]) # * ft2m # Left tip leading edge
point21 = np.array([13.400, -25.250, 7.617 + 0.1]) # * ft2m # Left tip trailing edge

do_plots=False

leading_edge_points = np.concatenate((np.linspace(point00, point10, int(num_wing_vlm/2+1))[0:-1,:], np.linspace(point10, point20, int(num_wing_vlm/2+1))), axis=0)
trailing_edge_points = np.concatenate((np.linspace(point01, point11, int(num_wing_vlm/2+1))[0:-1,:], np.linspace(point11, point21, int(num_wing_vlm/2+1))), axis=0)

leading_edge = wing.project(leading_edge_points, direction=np.array([-1., 0., 0.]), plot=do_plots)
trailing_edge = wing.project(trailing_edge_points, direction=np.array([1., 0., 0.]), plot=do_plots)


chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 0.5]), direction=np.array([0., 0., -1.]), grid_search_n=25, plot=do_plots, max_iterations=200)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 0.5]), direction=np.array([0., 0., 1.]), grid_search_n=25, plot=do_plots, max_iterations=200)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
# spatial_rep.plot_meshes([wing_camber_surface])



# wing beam mesh
num_wing_beam = 11
leading_edge = wing.project(leading_edge_points, direction=np.array([-1., 0., 0.]), plot=do_plots)
trailing_edge = wing.project(trailing_edge_points, direction=np.array([1., 0., 0.]), plot=do_plots)
wing_beam = am.linear_combination(leading_edge,trailing_edge,1,start_weights=np.ones((num_wing_beam,))*0.75,stop_weights=np.ones((num_wing_beam,))*0.25)
width = am.norm((leading_edge - trailing_edge)*0.5)
# width = am.subtract(leading_edge, trailing_edge)

if do_plots:
    spatial_rep.plot_meshes([wing_beam])

wing_beam = wing_beam.reshape((11,3))

offset = np.array([0,0,50])
top = wing.project(wing_beam.value+offset, direction=np.array([0., 0., -1.]), plot=do_plots)
bot = wing.project(wing_beam.value-offset, direction=np.array([0., 0., 1.]), plot=do_plots)
height = am.norm((top - bot)*1)
# height = am.subtract(top, bot)



# pass the beam meshes to aframe:
beam_mesh = LinearBeamMesh(
meshes = dict(
wing_beam = wing_beam,
wing_beam_width = width,
wing_beam_height = height,
)
)


# create the aframe dictionaries:
joints, bounds, beams = {}, {}, {}
beams['wing_beam'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'box','nodes': list(range(num_wing_beam))}
bounds['wing_root'] = {'beam': 'wing_beam','node': 5,'fdim': [1,1,1,1,1,1]}



# create the beam model:
beam = BeamM3LDisplacement(component=wing, mesh=beam_mesh, beams=beams, bounds=bounds, joints=joints)
beam.set_module_input('wing_beamt_cap_in', val=0.005, dv_flag=True, lower=0.001, upper=0.02, scaler=1E3)
beam.set_module_input('wing_beamt_web_in', val=0.005, dv_flag=True, lower=0.001, upper=0.02, scaler=1E3)







cruise_wing_structural_nodal_displacements_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
cruise_wing_aero_nodal_displacements_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_structural_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_aero_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh

order_u = 3
num_control_points_u = 35
knots_u_beginning = np.zeros((order_u-1,))
knots_u_middle = np.linspace(0., 1., num_control_points_u+2)
knots_u_end = np.ones((order_u-1,))
knots_u = np.hstack((knots_u_beginning, knots_u_middle, knots_u_end))
order_v = 1
knots_v = np.array([0., 0.5, 1.])

dummy_b_spline_space = lg.BSplineSpace(name='dummy_b_spline_space', order=(order_u,1), control_points_shape=((num_control_points_u,1)))
dummy_function_space = lg.BSplineSetSpace(name='dummy_space', spaces={'dummy_b_spline_space': dummy_b_spline_space})

cruise_wing_pressure_coefficients = m3l.Variable(name='cruise_wing_pressure_coefficients', shape=(num_control_points_u,3), value = np.zeros((num_control_points_u,3)))
cruise_wing_pressure = m3l.Function(name='cruise_wing_pressure', space=dummy_function_space, coefficients=cruise_wing_pressure_coefficients)

cruise_wing_displacement_coefficients = m3l.Variable(name='cruise_wing_displacement_coefficients', shape=(num_control_points_u,3))
cruise_wing_displacement = m3l.Function(name='cruise_wing_displacement', space=dummy_function_space, coefficients=cruise_wing_displacement_coefficients)

### Start defining computational graph ###

cruise_structural_wing_nodal_forces = cruise_wing_pressure(mesh=cruise_wing_structural_nodal_force_mesh)

beam_force_map_model = ebbeam.EBBeamForces(component=wing, beam_mesh=beam_mesh, beams=beams)
cruise_structural_wing_mesh_forces = beam_force_map_model.evaluate(nodal_forces=cruise_structural_wing_nodal_forces,
                                                                   nodal_forces_mesh=cruise_wing_structural_nodal_force_mesh)

beam_displacements_model = ebbeam.BeamM3LDisplacement(component=wing, mesh=beam_mesh, beams=beams, bounds=bounds, joints=joints)
residual = beam_displacements_model.evaluate(forces=cruise_structural_wing_mesh_forces)
beam_displacements_model.set_module_input('wing_beamt_cap_in', val=0.01, dv_flag=True, lower=0.0001, scaler=1E3)
beam_displacements_model.set_module_input('wing_beamt_web_in', val=0.01, dv_flag=True, lower=0.0001, scaler=1E3)

# beam_displacements_model = ebbeam.EBBeam(component=wing, mesh=beam_mesh, beams=beams, bounds=bounds, joints=joints)
# beam_displacements_model.set_module_input('wing_beamt_cap_in', val=0.01, dv_flag=True, lower=0.0001, scaler=1E3)
# beam_displacements_model.set_module_input('wing_beamt_web_in', val=0.01, dv_flag=True, lower=0.0001, scaler=1E3)

# cruise_structural_wing_mesh_displacements, cruise_structural_wing_mesh_rotations, wing_mass, wing_cg, wing_inertia_tensor = beam_displacements_model.evaluate(
#     forces=cruise_structural_wing_mesh_forces)

# beam_displacement_map_model = ebbeam.EBBeamNodalDisplacements(component=wing, beam_mesh=beam_mesh, beams=beams)
# cruise_structural_wing_nodal_displacements = beam_displacement_map_model.evaluate(beam_displacements=cruise_structural_wing_mesh_displacements,
#                                                                         nodal_displacements_mesh=cruise_wing_structural_nodal_displacements_mesh)

# test = cruise_structural_wing_nodal_displacements + cruise_structural_wing_nodal_forces

cruise_model = m3l.StructuralModalModel()
cruise_model.register_output(residual)
# cruise_model.register_output(cruise_structural_wing_nodal_displacements)



# m3l sizing model
# sizing_model = m3l.Model()
# sizing_model.register_output(wing_mass)
# sizing_model.register_output(wing_cg)
# sizing_model.register_output(wing_inertia_tensor)
# system_model.add_m3l_model('sizing_model', sizing_model)
# total_mass_properties = cd.TotalMassPropertiesM3L()
# total_mass, total_cg, total_inertia = total_mass_properties.evaluate(wing_mass, wing_cg, wing_inertia_tensor)
# cruise_model.register_output(total_mass)
# cruise_model.register_output(total_cg)
# cruise_model.register_output(total_inertia)



# design scenario
dc = cd.DesignScenario(name='struct')

# aircraft condition
# cruise condition
cruise_condition = cd.CruiseCondition(name="cruise_1")

cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()

cruise_condition.set_module_input(name='mach_number', val=0.17)
cruise_condition.set_module_input(name='range', val=40000)
cruise_condition.set_module_input(name='altitude', val=500)
cruise_condition.set_module_input(name='wing_incidence_angle', val=np.deg2rad(0))
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(2), dv_flag=False, lower=np.deg2rad(0), upper=np.deg2rad(5))
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 500]))

ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(ac_states)

cruise_condition.add_m3l_model('cruise_model', cruise_model)
dc.add_design_condition(cruise_condition)
system_model.add_design_scenario(dc)

caddee_csdl_model = caddee.assemble_csdl()


#caddee_csdl_model = cruise_model._assemble_csdl()


# testing_csdl_model.create_input('wing_beam_mesh', wing_beam.value.reshape((-1,3)))
# testing_csdl_model.create_input('wing_beam_width', width.value)
# testing_csdl_model.create_input('wing_beam_height', height.value)
# force_vector = np.zeros((num_wing_beam,3))
# force_vector[:,2] = 50000
# cruise_wing_forces = caddee_csdl_model.create_input('cruise_wing_pressure_input', val=force_vector)
# caddee_csdl_model.connect('cruise_wing_pressure_input', 'eb_beam_model.Aframe.wing_beam_forces') 



# caddee_csdl_model.add_objective('EulerEoMGenRefPt.trim_residual')


# caddee_csdl_model.add_constraint('system_model.struct.cruise_1.cruise_1.wing_eb_beam_model.Aframe.new_stress',upper=500E6/1,scaler=1E-8)
# caddee_csdl_model.add_objective('system_model.struct.cruise_1.cruise_1.total_constant_mass_properties.total_mass', scaler=1e-3)


# create and run simulator
sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()
# print(sim['system_model.struct.cruise_1.cruise_1.wing_eb_beam_model.Aframe.vm_stress'])
# print(sim['system_model.struct.cruise_1.cruise_1.wing_eb_beam_model.Aframe.wing_beam_forces'])
print(sim['system_model.struct.cruise_1.cruise_1.wing_eb_beam_model_displacement_jacobian_eig.e_real'])
print(sim['system_model.struct.cruise_1.cruise_1.wing_eb_beam_model_displacement_jacobian_eig.e_imag'])


# sim.compute_total_derivatives()
# sim.check_totals()


# prob = CSDLProblem(problem_name='lpc', simulator=sim)
# optimizer = SLSQP(prob, maxiter=1000, ftol=1E-6)
# optimizer.solve()
# optimizer.print_results()
