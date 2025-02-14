import numpy as np
import os
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from brian2.utils.logger import get_logger

class AbstractPyLifModelFloat(PyLoihiProcessModel):
    """Abstract implementation of floating point precision Leaky-Integrate-and-Fire neuron model.
    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = None  # OutPort of different LavaPyTypes
    v: np.ndarray = LavaPyType(np.ndarray, float)
    v_rs: float = LavaPyType(float, float)
    v_rev: float = LavaPyType(float, float)
    sigma_bg: float = LavaPyType(float, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    #bias: np.ndarray = LavaPyType(np.ndarray, float) # preparation for possible readout
    delta_v_ind: np.ndarray = LavaPyType(np.ndarray, float)

    def spiking_activation(self):
        """Abstract method to define the activation function that determines
        how spikes are generated.
        """
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Sub-threshold dynamics of postsynaptic potential and membrane voltage.
        """
        self.v = (self.v + activation_in) * (1 - self.delta_v_ind) + \
                 (self.v_rev + self.bias_mant) * self.delta_v_ind

    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; including reset of membrane voltage
        and starting of refractory period.
        """
        self.v[spike_vector] = self.v_rs

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        super().run_spk()
        a_in_data = self.a_in.recv()

        # Retrieve inputs for the current timestep
        if self.time_step < self.bias_for_all_times.shape[1]:
            self.bias_mant = self.bias_for_all_times[:, self.time_step]
            # Add noise
            if self.sigma_bg > self.EPSILON:
                rand = np.random.normal(loc=0.0, scale=1.0, size=self.proc_params._parameters["shape"],)
                self.bias_mant += self.sigma_bg * rand

        self.subthr_dynamics(activation_in=a_in_data)
        s_out_buff = self.spiking_activation()
        self.spiking_post_processing(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)


class AbstractPyLifModelFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision Leaky-Integrate-and-Fire neuron model.
    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: None  # OutPort of different LavaPyTypes
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v_rs: int = LavaPyType(int, np.int32, precision=17)
    v_rev: int = LavaPyType(int, np.int32, precision=17)
    sigma_bg: float = LavaPyType(int, np.int32, precision=17)
    delta_v_ind: np.ndarray = LavaPyType(np.ndarray, np.uint16, precision=12)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)
    #bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=16) # preparation for possible readout

    def __init__(self, proc_params):
        super(AbstractPyLifModelFixed, self).__init__(proc_params)
        # ds_offset and dm_offset are 1-bit registers in Loihi 1, which are
        # added to the delta_v_ind variable to compute effective decay constants
        # for postsynaptic potential and membrane voltage, respectively. They enable setting
        # decay constant values to exact 4096 = 2**12. Without them, the range of
        # 12-bit unsigned delta_j and delta_v_ind is 0 to 4095.
        self.ds_offset = 1
        self.dm_offset = 0
        self.effective_bias = 0
        # Let's define some bit-widths from Loihi
        # State variable v is 24-bits wide
        self.bitwidth = 24
        self.max_v_val = 2 ** (self.bitwidth - 1)
        # MSB alignment of decays by 12 bits
        # --> decay constants are accordingly prepared by Brian2Lava already
        self.decay_shift = 12
        self.decay_unity = 2**self.decay_shift

    def scale_bias(self):
        """Scale bias with bias exponent by taking into account sign of the
        exponent.
        """
        # Create local copy of bias_mant with promoted dtype to prevent
        # overflow when applying shift of bias_exp.
        bias_mant = self.bias_mant.copy().astype(np.int32)
        self.effective_bias = np.where(
            self.bias_exp >= 0,
            np.left_shift(bias_mant, self.bias_exp),
            np.right_shift(bias_mant, -self.bias_exp),
        )

    def scale_threshold(self):
        """Placeholder method for scaling threshold(s)."""
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def spiking_activation(self):
        """Placeholder method to specify spiking behaviour of a LIF neuron."""
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Sub-threshold dynamics of postsynaptic potential and membrane voltage.
        """
        neg_v_limit = -np.int32(self.max_v_val) + 1
        pos_v_limit = np.int32(self.max_v_val) - 1
        # Update membrane voltage (much simplified compared to 'lif_rp_v_input')
        # ----------------------------------------------------------------------
        decay_const_v = self.delta_v_ind + self.dm_offset
        v_decayed = np.int64(self.v + activation_in) * (self.decay_unity - decay_const_v)
        v_decayed = np.sign(v_decayed) * np.right_shift(
        	np.abs(v_decayed), self.decay_shift
        )
        v_increase = np.int64(self.v_rev + self.effective_bias) * decay_const_v
        v_increase = np.sign(v_increase) * np.right_shift(
        	np.abs(v_increase), self.decay_shift
        )
        v_updated = np.int32(v_decayed + v_increase)
        self.v = np.clip(v_updated, neg_v_limit, pos_v_limit)

    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; including reset of membrane voltage
        and starting of refractory period.
        """
        self.v[spike_vector] = self.v_rs

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """        
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        # Retrieve inputs for the current timestep
        if self.time_step < self.bias_for_all_times.shape[1]:
            self.bias_mant = self.bias_for_all_times[:, self.time_step]
            # Add noise
            if self.sigma_bg > self.EPSILON:
                rand = np.random.normal(loc=0.0, scale=1.0, size=self.proc_params._parameters["shape"],)
                self.bias_mant += self.sigma_bg * rand

        # Compute effective bias
        self.scale_bias()

        # Compute subthreshold and spiking dynamics
        self.subthr_dynamics(activation_in=a_in_data)
        s_out_buff = self.spiking_activation()

        # Do post-processing
        self.spiking_post_processing(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)

@implements(proc=LIF_predef_stim_versatile, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifModelFloat(AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v_th: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super(PyLifModelFloat, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyLifModelFloat process model "
                          f"(shape: {self.proc_params._parameters['shape']})")

        # load bias values for all times from numpy file (which should have been stored by Brian2Lava)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bias_for_all_times = np.load(os.path.join(script_dir, "bias_for_all_times.npy"))
        self.logger.debug(f"Loaded bias values for all timesteps, shape: {self.bias_for_all_times.shape}")

        # set epsilon
        self.EPSILON = 1e-7

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.v_th


@implements(proc=LIF_predef_stim_versatile, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifModelFixed(AbstractPyLifModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process in fixed point
    precision to mimic the behavior of Loihi 2 bit-by-bit.

    Precisions of state variables

    - delta_v_ind: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - v_th: unsigned 17-bit integer (0 to 131071).
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    v_th: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyLifModelFixed, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyLifModelFixed process model "
                          f"(shape: {self.proc_params._parameters['shape']})")

        # load bias values for all times from numpy file (which should have been stored by Brian2Lava)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bias_for_all_times = np.load(os.path.join(script_dir, "bias_for_all_times.npy"))
        self.logger.debug(f"Loaded bias values for all timesteps, shape: {self.bias_for_all_times.shape}")

        # set epsilon
        self.EPSILON = 1e-7

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.v_th
