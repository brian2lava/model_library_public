import numpy as np
import typing as ty
from brian2.utils.logger import get_logger
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

@implements(proc=ProbSpiker, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyProbSpikerModelFloat(PyLoihiProcessModel):
    """Implementation of floating-point precision
    probabilistic spiker model.
    """
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    rnd: np.ndarray = LavaPyType(np.ndarray, float)
    p_spike: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super(PyProbSpikerModelFloat, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyProbSpikerModelFloat process model")

    def spiking_activation(self):
        """Spiking activation function."""
        self.rnd = np.random.rand(self.shape[0])
        return self.rnd < self.p_spike

    def run_spk(self):
        """The run function that performs the actual computation."""
        self.s_out.send(self.spiking_activation())

@implements(proc=ProbSpiker, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyProbSpikerModelFixed(PyLoihiProcessModel):
    """Implementation of fixed-point precision
    probabilistic spiker model.
    """
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    rnd: np.ndarray = LavaPyType(np.ndarray, int)
    p_spike: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super(PyProbSpikerModelFixed, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyProbSpikerModelFixed process model")

        # MSB alignment of random numbers by 24 bits
        # --> probability is accordingly prepared by Brian2Lava 
        self.random_unity = 2**24

    def spiking_activation(self):
        """Spiking activation function."""
        self.rnd = np.random.randint(0, self.random_unity+1, size=self.shape[0])
        return self.rnd < self.p_spike

    def run_spk(self):
        """The run function that performs the actual computation."""
        self.s_out.send(self.spiking_activation())
