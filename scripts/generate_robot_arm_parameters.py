import numpy as np
import time
from casadi import * 

class robot_arm_params: 

    def __init__(self, robot_arm_length, mass_distribution, time_step, alpha, id): 

        self._robot_arm_length = robot_arm_length
        self._mass_distribution = mass_distribution
        self._created = 0 
        self._time_step = time_step
        self._alpha = alpha
        self._id = id

    def from_custom(self, Kse, Kbt):

        self._Kse = Kse 
        self._Kbt = Kbt 

    def from_hollow_rod(self): 

        pass 

    def from_solid_rod(self, r, shear_mod, E, rho):

        area = np.pi * r**2
        I = (np.pi * r**4) / 4
        J = 2 * I
        self._Kse = diag([shear_mod * area,
                        shear_mod * area, E * area])
        self._Kbt = diag([E * I, E *
                        I, shear_mod * J])
        self._mass_distribution = rho * area
        self._rho = rho
        self._J = J 
        self._I = I

    def set_damping_coefficient(self, C): 

        self._C = C 

    def set_damping_factor(self, Bbt, Bse): 

        self._Bbt = Bbt
        self._Bse = Bse

    def set_rotational_inertia_matrix(self, J): 

        self._J = J

    def set_rho(self, rho): 

        self._rho = rho

    def get_id(self):

        return self._id

    def get_J(self): 

        return self._J

    def get_Bbt(self): 

        return self._Bbt

    def get_Bse(self): 

        return self._Bse

    def get_arm_length(self): 

        return self._robot_arm_length 

    def get_mass_distributions(self): 

        return self._mass_distributions 

    def get_Kse(self):

        return self._Kse

    def get_Kbt(self): 

        return self._Kbt

    def get_time_step(self): 

        return self._time_step

    def get_alpha(self): 

        return self._alpha

    def get_C(self): 

        return self._C 

    def get_mass_distribution(self): 

        return self._mass_distribution

    def get_rho(self): 

        return self._rho
