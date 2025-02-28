import numpy as np
import typing as ty

from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.neuron import LearningNeuronProcess
from brian2.utils.logger import get_logger

class AbstractLIF(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        v: ty.Union[float, list, np.ndarray],
        delta_v_ind: float,
        bias_mant: ty.Union[float, list, np.ndarray],
        bias_exp: ty.Union[float, list, np.ndarray],
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            v=v,
            delta_v_ind=delta_v_ind,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )
        self.logger = get_logger('brian2.devices.lava')

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=v)
        self.delta_v_ind = Var(shape=shape, init=delta_v_ind)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.bias_mant = Var(shape=shape, init=bias_mant)


class LIF_delta_v_input_v_rev_tau_v_ind(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with refractory period and
    delta-shaped postsynaptic potential input.

    LIF dynamics abstracts to:
    v[t] = v[t-1] * (1-delta_v_ind) + (a_in + bias) * delta_v_ind    # neuron voltage
    s_out = v[t] > v_th                                      # spike if threshold is exceeded
    v[t] = 0                                                 # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    delta_v_ind : float, optional
        Inverse of decay time constant `tau_v` for voltage decay. Currently, 
        only a single decay can be set for the entire population of neurons.
    delta_v_ind : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    v_th : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    v_rs : float, optional
        Neuron reset voltage after spike.
    v_rev : float, optional
        Neuron reversal voltage.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), v_th=5)
    This will create 200x15 LIF neurons that all have the same threshold voltage of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        delta_v_ind: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v_th: ty.Optional[float] = 100,
        v_rs: ty.Optional[float] = 0,
        v_rev: ty.Optional[float] = 0,
        #bias: ty.Optional[ty.Union[float, list, np.ndarray]] = 0, # preparation for possible readout
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs) -> None:
        super().__init__(
            shape=shape,
            v=v,
            delta_v_ind=delta_v_ind,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )
        # Set threshold and reset voltage
        self.v_th = Var(shape=(1,), init=v_th)
        self.v_rs = Var(shape=(1,), init=v_rs)
        self.v_rev = Var(shape=(1,), init=v_rev)
        #self.bias = Var(shape=shape, init=0)
        msg_var_par = f"Initialized attributes in process '{self.name}'"
            
        # Print the values
        msg_var_par = f"""{msg_var_par}:
             shape = {shape}
             v = {v}
             delta_v_ind = {delta_v_ind} (computed from tau_v_ind)
             bias_mant = {self.bias_mant.init}, bias_exp = {self.bias_exp.init}
             v_th = {self.v_th.init}
             v_rs = {self.v_rs.init}
             v_rev = {self.v_rev.init}"""
        self.logger.debug(msg_var_par)
        
