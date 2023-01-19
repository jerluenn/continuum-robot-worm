from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
import time

class robot_arm_model: 

    def __init__(self, _robot_params): 

        self._robot_params = _robot_params
        self._integration_steps = 10
        self._integration_stages = 4
        self._build_robot_arm() 

    def _build_robot_arm(self): 

        self._initialiseStates()
        self._createIntegrator()
        self._createStepIntegrator() 

    def _initialiseStates(self): 

        # Initialise states

        self._growth_rate = 1.0
        self._p = SX.sym('p', 3)
        self._R = SX.sym('R', 9)
        self._n = SX.sym('n', 3)
        self._m = SX.sym('m', 3)
        self._q = SX.sym('q', 3)
        self._om = SX.sym('om', 3)
        self._tau = SX.sym('tau', 3)

        # Initialise constants

        """TO DO: Update c0"""

        self._c0 = (1.5 + self._robot_params.get_alpha())/(self._robot_params.get_time_step()*(1 + self._robot_params.get_alpha()))

        self._g = SX([9.81, 0, 0])
        self._f_ext = self._robot_params.get_mass_distribution() * self._g

        # Intermediate states
        self._v_history = SX.sym('v_hist', 3)
        self._u_history = SX.sym('u_hist', 3)
        self._q_history = SX.sym('q_hist', 3)
        self._om_history = SX.sym('om_hist', 3)

        self._u = inv(self._robot_params.get_Kbt() + self._c0*self._robot_params.get_Bbt())@(transpose(reshape(self._R, 3, 3))@self._m - self._robot_params.get_Bbt()@self._u_history)
        self._v = inv(self._robot_params.get_Kse() + self._c0*self._robot_params.get_Bse())@(transpose(reshape(self._R, 3, 3))@self._n + self._robot_params.get_Kse()@SX([0, 0, 1]) - self._robot_params.get_Bse()@self._v_history)

        self._v_t = self._c0*self._v + self._v_history
        self._u_t = self._c0*self._u + self._u_history
        self._q_t = self._c0*self._q + self._q_history
        self._om_t = self._c0*self._om + self._om_history

        # Distributed forces and moments

    def _createModel(self): 

        p_dot = reshape(self._R, 3, 3) @ self._v
        R_dot = reshape(self._R, 3, 3) @ skew(self._u)

        # n_dot = reshape(self._R, 3, 3) @ (self._robot_params.get_mass_distribution()*(skew(self._om)@self._q + self._q_t)) - self._robot_params.get_mass_distribution()*self._g + self._robot_params.get_C()@(self._q*self._q*(1/(1 + SX.exp(-self._growth_rate*self._q)))) - self.get_external_distributed_forces()
        n_dot = - self._robot_params.get_mass_distribution()*self._g - self.get_external_distributed_forces()
        # m_dot = self._robot_params.get_rho() * reshape(self._R, 3, 3) @ (skew(self._om) @ self._robot_params.get_J() @ self._om + self._robot_params.get_J()@self._om_t) - skew(p_dot)@self._n
        m_dot = - skew(p_dot)@self._n
        q_dot = self._v_t - skew(self._u)@self._q + skew(self._om)@self._v
        om_dot = self._u_t - skew(self._u)@self._om

        tau_dot = SX.zeros(self._tau.shape[0])

        xdot = vertcat(p_dot, reshape(R_dot, 9, 1), n_dot, m_dot, q_dot, om_dot, tau_dot)

        return xdot

    def get_external_distributed_forces(self):

        """TO DO: define p_dot and p_dotdot"""

        p_dot = reshape(self._R, 3, 3) @ self._v

        p_dotdot = reshape(self._R, 3, 3) @ skew(self._u) @ self._v

        for i in range(self._tau.shape[0]):

            f_t = - (self._tau[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 

    def _createIntegrator(self):  

        model_name = 'robot_model_' + self._robot_params.get_id()

        """TO DO: 
            1. Set x.
            2. Set xdot.
            3. Set p.

        """

        x = vertcat(self._p, self._R, self._n, self._m, self._q, self._om, self._tau)
        p = vertcat(self._v_history, self._u_history, self._q_history, self._om_history) 
        xdot = self._createModel()

        self._model = AcadosModel()
        self._model.name = model_name
        self._model.x = x
        self._model.u = SX([])
        self._model.f_expl_expr = xdot
        self._model.p = p 
        self._model.z = []

        sim = AcadosSim()

        sim.parameter_values = np.zeros(12) #Number of parameters that need to be updated for CR dynamics 
        sim.model = self._model
        Sf = self._robot_params.get_arm_length()

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = self._integration_steps
        # sim.solver_options.sens_forw = False 
        sim.solver_options.sens_forw = True

        self._integrator = AcadosSimSolver(sim)

        return self._integrator

    def _createStepIntegrator(self): 

        model_name = 'robot_model_step_' + self._robot_params.get_id()

        """TO DO: 
            1. Set x.
            2. Set xdot.
            3. Set p.

        """

        x = vertcat(self._p, self._R, self._n, self._m, self._q, self._om, self._tau)
        p = vertcat(self._v_history, self._u_history, self._q_history, self._om_history) 
        xdot = self._createModel()

        self._model = AcadosModel()
        self._model.name = model_name
        self._model.x = x
        self._model.u = SX([])
        self._model.f_expl_expr = xdot
        self._model.p = p 
        self._model.z = []

        sim = AcadosSim()

        sim.parameter_values = np.zeros(12) #Number of parameters that need to be updated for CR dynamics
        sim.model = self._model
        Sf = self._robot_params.get_arm_length()

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = 1
        sim.solver_options.sens_forw = False 
        # sim.solver_options.sens_forw = True

        self._stepIntegrator = AcadosSimSolver(sim)

        return self._stepIntegrator
        

    def set_num_integration_stages(self, stages):

        self._integration_stages = stages

    def set_num_integrator_steps(self, steps): 

        self._integration_steps = steps

    def get_num_integration_stages(self):

        return self._integration_stages

    def get_num_integration_steps(self):

        return self._integration_steps

    def get_robot_params(self):

        return self._robot_params

    def get_model(self): 

        return self._model

    


""" 

TO DO: 

1. Create new params for robot arms with springs. 
2. Create new model: 
    2a. Add parameters to account for semi-discretization terms. 
    2b. Change ODE to account for extensibility. 
    2c. Create new states. 

"""