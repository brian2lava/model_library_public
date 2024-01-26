import numpy as np
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
    s_out = None  # This will be an OutPort of different LavaPyTypes
    v_psp: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    v_rs: float = LavaPyType(float, float)
    t_rp_steps: int = LavaPyType(int, int)
    t_rp_steps_end: np.ndarray = LavaPyType(np.ndarray, int) # This indicates until which timestep a neuron is in refractory period
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    delta_psp: float = LavaPyType(float, float)
    delta_v: float = LavaPyType(float, float)

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
        self.v_psp[:] = self.v_psp * (1 - self.delta_psp) + activation_in

        non_ref = self.t_rp_steps_end < self.time_step
        self.v[non_ref] = self.v[non_ref] * (1 - self.delta_v) + self.v_psp[non_ref] + self.bias_mant[non_ref]

    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; including reset of membrane voltage
        and starting of refractory period.
        """
        self.v[spike_vector] = self.v_rs
        self.t_rp_steps_end[spike_vector] = (self.time_step + self.t_rp_steps)

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        super().run_spk()
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)
        self.s_out_buff = self.spiking_activation()
        self.spiking_post_processing(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)


class AbstractPyLifModelFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision Leaky-Integrate-and-Fire neuron model.
    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: None  # This will be an OutPort of different LavaPyTypes
    v_psp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v_rs: int = LavaPyType(int, np.int32, precision=17)
    t_rp_steps: int = LavaPyType(int, int)
    t_rp_steps_end: np.ndarray = LavaPyType(np.ndarray, int) # This indicates until which timestep a neuron is 
                                                             # in refractory period
    delta_psp: int = LavaPyType(int, np.uint16, precision=12)
    delta_v: int = LavaPyType(int, np.uint16, precision=12)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)

    def __init__(self, proc_params):
        super(AbstractPyLifModelFixed, self).__init__(proc_params)
        # ds_offset and dm_offset are 1-bit registers in Loihi 1, which are
        # added to delta_psp and delta_v variables to compute effective decay constants
        # for postsynaptic potential and membrane voltage, respectively. They enable setting
        # decay constant values to exact 4096 = 2**12. Without them, the range of
        # 12-bit unsigned delta_psp and delta_v is 0 to 4095.
        self.ds_offset = 0 #1 # TODO fix? this has been causing problems
        self.dm_offset = 0
        self.effective_bias = 0
        # Let's define some bit-widths from Loihi
        # State variables v_psp and v are 24-bits wide
        self.bitwidth = 24
        self.max_val = 2 ** (self.bitwidth - 1)
        # MSB alignment of decays by 12 bits
        # --> decay constants are accordingly prepared by Brian2Lava 
        self.decay_shift = 12
        self.decay_unity = 2**self.decay_shift
        # MSB alignment of incoming activation and voltage parameters by 6 bits
        # --> constants are accordingly prepared by Brian2Lava 
        self.act_unity = 2**6

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
        # Update postsynaptic potential
        # -----------------------------
        # Compute decay constant (left shift via multiplication by `decay_unity`
        # --> already done by Brian2Lava!)
        #decay_const_psp = self.delta_psp*self.decay_unity + self.ds_offset
        decay_const_psp = self.delta_psp + self.ds_offset
        # Below, v_psp is promoted to int64 to avoid overflow of the product
        # between v_psp and decay term beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits)
        v_psp_decayed = np.int64(self.v_psp) * (self.decay_unity - decay_const_psp)
        v_psp_decayed = np.sign(v_psp_decayed) * np.right_shift(
        	np.abs(v_psp_decayed), self.decay_shift
        )
        # Hardware left-shifts synaptic input for MSB alignment
        activation_in = activation_in * self.act_unity
        # Add synaptic input to decayed postsynaptic potential
        v_psp_updated = np.int32(v_psp_decayed + activation_in)
        # Check if value of postsynaptic potential is within bounds of 24-bit. Overflows are
        # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
        # becomes k and -(2**23 + k) becomes -k
        wrapped_psp = np.where(
        	v_psp_updated > self.max_val,
        	v_psp_updated - 2 * self.max_val,
        	v_psp_updated,
        )
        wrapped_psp = np.where(
        	wrapped_psp <= -self.max_val,
        	v_psp_updated + 2 * self.max_val,
        	wrapped_psp,
        )
        self.v_psp[:] = wrapped_psp

        # Update membrane voltage (decay similar to postsynaptic potential)
        # -----------------------------------------------------------------
        decay_const_v = self.delta_v + self.dm_offset
        neg_voltage_limit = -np.int32(self.max_val) + 1
        pos_voltage_limit = np.int32(self.max_val) - 1
        v_decayed = np.int64(self.v) * (self.decay_unity - decay_const_v)
        v_decayed = np.sign(v_decayed) * np.right_shift(
        	np.abs(v_decayed), self.decay_shift
        )
        v_updated = np.int32(v_decayed + self.v_psp + self.effective_bias)
        non_ref = self.t_rp_steps_end < self.time_step
        self.v[non_ref] = np.clip(v_updated[non_ref], neg_voltage_limit, pos_voltage_limit)

    def spiking_post_processing(self, spike_vector: np.ndarray):
        """Post processing after spiking; including reset of membrane voltage
        and starting of refractory period.
        """
        self.v[spike_vector] = self.v_rs
        self.t_rp_steps_end[spike_vector] = (self.time_step + self.t_rp_steps)

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """        
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        # Compute effective bias
        self.scale_bias()

        # Compute subthreshold and spiking dynamics
        self.subthr_dynamics(activation_in=a_in_data)
        self.s_out_buff = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.spiking_post_processing(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)

@implements(proc=LIF_rp_v_input, protocol=LoihiProtocol)
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
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyLifModelFloat process model")

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.v_th


@implements(proc=LIF_rp_v_input, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifModelFixed(AbstractPyLifModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process in fixed point
    precision to mimic the behavior of Loihi 2 bit-by-bit.

    Precisions of state variables

    - delta_psp: unsigned 12-bit integer (0 to 4095)
    - delta_v: unsigned 12-bit integer (0 to 4095)
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
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyLifModelFixed process model")

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.v_th
