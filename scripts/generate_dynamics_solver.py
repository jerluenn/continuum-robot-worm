from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np

import time
import generate_robot_arm_model
from matplotlib import pyplot as plt

class dynamics_solver:

    def __init__(self, _robot_arm_model): 

        self._robot_arm_model = _robot_arm_model
        self._boundary_length = self._robot_arm_model.get_robot_params().get_arm_length()
        self._integration_steps = self._robot_arm_model.get_num_integration_steps()
        self.createHorizontalSolver()

    """Create solver based on multiple scenarios?"""

    def createHorizontalSolver(self): 

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_model()
        nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = nx + nu

        self.ocp.dims.N = self._integration_steps
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = np.identity(nx)
        self.ocp.cost.W = np.zeros((ny, ny))
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx_e = np.zeros((nx, nx))
        self.ocp.cost.Vx_e[12:18, 12:18] = np.identity(6)
        self.ocp.cost.yref  = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.solver_options.qp_solver_iter_max = 400
        # self.ocp.solver_options.sim_method_num_steps = self.integration_steps
        self.ocp.parameter_values = np.zeros(12)
        self.ocp.solver_options.qp_solver_warm_start = 2

        # self.ocp.solver_options.levenberg_marquardt = 0.1

        self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0
        self.ocp.solver_options.regularize_method = 'CONVEXIFY'

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length

        wrench_ub = 20
        pos_max = 5
        R_max = 1.005
        q_max = 1.0
        om_max = 1.0

        self.ocp.constraints.idxbx_0 = np.arange(27)

        self.ocp.constraints.lbx_0 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub, 0, 0, 0, 0, 0, 0, 0, 0, 0])  
        # p, R, n, m, q, om, tau.

        self.ocp.constraints.ubx_0 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        # p, R, n, m, q, om, tau.

        # self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        self.ocp.constraints.idxbx = np.arange(27)

        self.ocp.constraints.lbx = np.hstack((-pos_max, -pos_max, -pos_max, -R_max, -R_max, -R_max, -R_max, -R_max, -R_max, -R_max, -R_max, -R_max, -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub, -300*np.ones(6), -30*np.ones(3)))
        self.ocp.constraints.ubx = -self.ocp.constraints.lbx

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        name = self.ocp.model.name + 'dynamics_solver'
        self.ocp.code_export_directory = name

        self._horizontal_solver = AcadosOcpSolver(self.ocp, json_file=f'{name}.json')
        self._horizontal_integrator = AcadosSimSolver(self.ocp, json_file=f'{name}.json')

        return self._horizontal_solver, self._horizontal_integrator

    def createPendulumTypeSolver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_model()
        nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = nx + nu

        self.ocp.dims.N = self.integration_steps
        # self.ocp.cost.cost_type_0 = 'LINEAR_LS'
        # self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = np.identity(nx)
        self.ocp.cost.W = np.zeros((ny, ny))
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx_e = np.zeros((nx, nx))
        self.ocp.cost.Vx_e[0:7, 0:7] = np.identity(7)
        self.ocp.cost.yref  = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.solver_options.qp_solver_iter_max = 400
        # self.ocp.solver_options.sim_method_num_steps = self.integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.1

        # self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0
        self.ocp.solver_options.regularize_method = 'CONVEXIFY'

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self.boundary_length
        

        wrench_ub = -5
        wrench_ub = 5

        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        self.ocp.constraints.lbx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub,
            0])  # tension, alpha, kappa, curvature

        self.ocp.constraints.ubx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub,
            0]) 

        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        self.ocp.constraints.lbx = np.array([-5, -5, -5, -1.05, -1.05, -1.05, -1.05, #pose at start.
            -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub , -wrench_ub, 
            0])

        self.ocp.constraints.ubx = np.array([5, 5, 5, 1.05, 1.05, 1.05, 1.05, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub, 
            50])

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator

    def get_horizontal_solver(self): 

        return self._horizontal_solver

    def get_horizontal_integrator(self): 

        return self._horizontal_integrator
