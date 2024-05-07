import numpy as np
import typing as ty
from brian2.utils.logger import get_logger
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

@implements(proc=TimeSpiker, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyTimeSpikerModelFloat(PyLoihiProcessModel):
    """Implementation of floating-point precision
    time-specific spiker model.
    """
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    t_rp_steps: int = LavaPyType(int, int)
    t_rp_steps_end: np.ndarray = LavaPyType(np.ndarray, int)
    t_spike_steps: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super(PyTimeSpikerModelFloat, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyTimeSpikerModelFloat process model")

    def spiking_activation(self):
        """Spiking activation function."""
        non_ref = self.t_rp_steps_end < self.time_step
        return np.logical_and(self.time_step > self.t_spike_steps, non_ref)
    
    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; starting of refractory period.
        """
        self.t_rp_steps_end[spike_vector] = (self.time_step + self.t_rp_steps)

    def run_spk(self):
        """The run function that performs the actual computation."""
        _ = self.a_in.recv()
        s_out_buff = self.spiking_activation()
        self.spiking_post_processing(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)

@implements(proc=TimeSpiker, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyTimeSpikerModelFixed(PyLoihiProcessModel):
    """Implementation of fixed-point precision
    time-specific spiker model.
    """
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    t_rp_steps: int = LavaPyType(int, int)
    t_rp_steps_end: np.ndarray = LavaPyType(np.ndarray, int)
    t_spike_steps: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super(PyTimeSpikerModelFixed, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyTimeSpikerModelFixed process model")

    def spiking_activation(self):
        """Spiking activation function."""
        #return np.zeros(self.shape)
        non_ref = self.t_rp_steps_end < self.time_step
        return np.logical_and(self.time_step > self.t_spike_steps, non_ref)
    
    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; starting of refractory period.
        """
        self.t_rp_steps_end[spike_vector] = (self.time_step + self.t_rp_steps)

    def run_spk(self):
        """The run function that performs the actual computation."""
        _ = self.a_in.recv()
        s_out_buff = self.spiking_activation()
        self.spiking_post_processing(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)
