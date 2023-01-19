from generate_robot_arm_model import robot_arm_model
from generate_robot_arm_parameters import robot_arm_params
from generate_dynamics_solver import dynamics_solver

import numpy as np 

NUM_ITERATIONS = 1000

robot_arm_1 = robot_arm_params(0.15, 0.03, 0.01, -0.5, "1")
robot_arm_1.from_solid_rod(0.001, 100e9, 200e9, 8000)
C = np.diag([0.03, 0.03, 0.03])
Bbt = np.diag([1e-6, 1e-6, 1e-6])
Bse = Bbt
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)

robot_arm_model_1 = robot_arm_model(robot_arm_1)
d1 = dynamics_solver(robot_arm_model_1)

yref = np.zeros(21)

d1.get_horizontal_solver().cost_set(robot_arm_model_1.get_num_integration_steps(), 'yref', yref)

initial_solution = np.zeros(21)
initial_solution[3] = 1
initial_solution[7] = 1
initial_solution[11] = 1
initial_solution[12] = 3.69828287e-02
initial_solution[16] = 0.0027
d1.get_horizontal_solver().set(0, 'x', initial_solution)

subseq_solution = initial_solution

for i in range(robot_arm_model_1.get_num_integration_steps()): 

    d1.get_horizontal_integrator().set('x', subseq_solution)
    d1.get_horizontal_integrator().solve()
    subseq_solution = d1.get_horizontal_integrator().get('x')
    d1.get_horizontal_solver().set(i+1, 'x', subseq_solution)

a = 1

for i in range(NUM_ITERATIONS): 

    d1.get_horizontal_solver().solve()
    print(d1.get_horizontal_solver().get_residuals())
    print(d1.get_horizontal_solver().get_cost())

# """TO DO: FIX CONSTRAINTS."""

# d1.get_horizontal_solver().solve()





