import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from brian2.utils.logger import get_logger

class AbstractPyLifModelFloat(PyLoihiProcessModel):
    """Abstract implementation of floating point precision
    leaky-integrate-and-fire neuron model.

    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = None  # This will be an OutPort of different LavaPyTypes
    j: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    v_rs: float = LavaPyType(float, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    delta_j: float = LavaPyType(float, float)
    delta_v: float = LavaPyType(float, float)
    dt: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Abstract method to define the activation function that determines
        how spikes are generated.
        """
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """
        self.j[:] = self.j * (1 - self.delta_j) + activation_in
        
        self.v[:] = self.v * (1 - self.delta_v) + \
                    (self.j + self.bias_mant) * self.dt

    def reset_voltage(self, spike_vector: np.ndarray):
        """Voltage reset behaviour. This can differ for different neuron
        models."""
        self.v[spike_vector] = self.v_rs

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        super().run_spk()
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)
        s_out_buff = self.spiking_activation()
        self.reset_voltage(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)


class AbstractPyLifModelFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision
    leaky-integrate-and-fire neuron model. Implementations like those
    bit-accurate with Loihi hardware inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: None  # This will be an OutPort of different LavaPyTypes
    j: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    delta_j: int = LavaPyType(int, np.uint16, precision=12)
    delta_v: int = LavaPyType(int, np.uint16, precision=12)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)
    v_rs: int = LavaPyType(int, np.int32, precision=17)
    dt: int = LavaPyType(int, np.uint16)

    def __init__(self, proc_params):
        super(AbstractPyLifModelFixed, self).__init__(proc_params)
        # ds_offset and dm_offset are 1-bit registers in Loihi 1, which are
        # added to delta_j and delta_v variables to compute effective decay constants
        # for current and voltage, respectively. They enable setting decay
        # constant values to exact 4096 = 2**12. Without them, the range of
        # 12-bit unsigned delta_j and delta_v is 0 to 4095.
        self.ds_offset = 1
        self.dm_offset = 0
        self.effective_bias = 0
        # Let's define some bit-widths from Loihi
        # State variables j and v are 24-bits wide
        self.jv_bitwidth = 24
        self.max_jv_val = 2 ** (self.jv_bitwidth - 1)
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

    def reset_voltage(self):
        """Placeholder method for voltage reset."""
        raise NotImplementedError(
            "reset_voltage() cannot be called from "
            "an abstract ProcessModel"
        )

    def spiking_activation(self):
        """Placeholder method to specify spiking behaviour of a LIF neuron."""
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """
        # Update current
        # --------------
        # Compute decay constant. Left shift of `delta_j` by `log2(decay_unity)`` is
        # already done by Brian2Lava! If `ds_offset > 0`, clip to ensure that it doesn't 
        # exceed `decay_unity`.
        decay_const_j = np.clip(self.delta_j + self.ds_offset, 0, self.decay_unity)
        # Below, j is promoted to int64 to avoid overflow of the product
        # between j and decay term beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits).
        j_decayed = np.int64(self.j) * (self.decay_unity - decay_const_j) 
        j_decayed = np.sign(j_decayed) * np.right_shift(
        	np.abs(j_decayed), self.decay_shift
        )
        # Add synaptic input to decayed postsynaptic current
        j_updated = np.int32(j_decayed + activation_in)
        # Check if value of current is within bounds of 24-bit. Overflows are
        # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
        # becomes k and -(2**23 + k) becomes -k
        #wrapped_curr = np.where(
        #	j_updated > self.max_jv_val,
        #	j_updated - 2 * self.max_jv_val,
        #	j_updated,
        #)
        #wrapped_curr = np.where(
        #	wrapped_curr <= -self.max_jv_val,
        #	j_updated + 2 * self.max_jv_val,
        #	wrapped_curr,
        #)
        #self.j[:] = wrapped_curr
        # --> Instead of wrapping, we better do clipping (as for voltage
        # below)
        neg_jv_limit = -np.int32(self.max_jv_val) + 1
        pos_jv_limit = np.int32(self.max_jv_val) - 1
        self.j[:] = np.clip(j_updated, neg_jv_limit, pos_jv_limit)

        # Update voltage (decay similar to current)
        # -----------------------------------------
        decay_const_v = self.delta_v + self.dm_offset
        v_decayed = np.int64(self.v) * (self.decay_unity - decay_const_v)
        v_decayed = np.sign(v_decayed) * np.right_shift(
        	np.abs(v_decayed), self.decay_shift
        )
        v_increase = np.int64(self.j + self.effective_bias) * self.dt * self.decay_unity
        v_increase = np.sign(v_increase) * np.right_shift(
            np.abs(v_increase), self.decay_shift
        )
        v_updated = np.int32(v_decayed + v_increase)
        self.v[:] = np.clip(v_updated, neg_jv_limit, pos_jv_limit)


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
        s_out_buff = self.spiking_activation()

        # Do post-processing
        self.reset_voltage(spike_vector=s_out_buff)
        self.s_out.send(s_out_buff)


@implements(proc=LIF, protocol=LoihiProtocol)
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
        return self.v >= self.v_th


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifModelBitAcc(AbstractPyLifModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process bit-accurate
    with Loihi's hardware LIF dynamics, which means, it mimics Loihi
    behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - delta_j: unsigned 12-bit integer (0 to 4095)
    - delta_v: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - v_th: unsigned 17-bit integer (0 to 131071).
    - v_rs: unsigned 17-bit integer (0 to 131071).
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    v_th: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyLifModelBitAcc, self).__init__(proc_params)
        self.logger = get_logger('brian2.devices.lava')
        self.logger.debug(f"Process '{proc_params._parameters['name']}' initialized with PyLifModelBitAcc process model")

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v >= self.v_th

    def reset_voltage(self, spike_vector: np.ndarray):
        """Voltage reset behavior.
        """
        self.v[spike_vector] = self.v_rs