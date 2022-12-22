from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
import time

class robot_arm: 

    def __init__(self, _params): 

        self._params = _params
        self._build_robot_arm() 

    def _build_robot_arm(self): 

        self._initialiseStates()
        self._createIntegrator()
        self._createstepIntegrator() 

    def _initialiseStates(self): 

        pass 

    def _createIntegrator(self):  

        pass 

    def _createStepIntegrator(self): 

        pass 


""" 

TO DO: 

1. Create new params for robot arms with springs. 
2. Create new model: 
    2a. Add parameters to account for semi-discretization terms. 
    2b. Change ODE to account for extensibility. 
    2c. Create new states. 

"""