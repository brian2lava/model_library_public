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
        u: ty.Union[float, list, np.ndarray],
        v: ty.Union[float, list, np.ndarray],
        du: float,
        dv: float,
        bias_mant: ty.Union[float, list, np.ndarray],
        bias_exp: ty.Union[float, list, np.ndarray],
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )
        self.logger = get_logger('brian2.devices.lava')

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=u)
        self.v = Var(shape=shape, init=v)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.bias_mant = Var(shape=shape, init=bias_mant)


class LIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    vrs : float, optional
        Neuron reset voltage after spike.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), du=10, dv=5)
    This will create 200x15 LIF neurons that all have the same current decay
    of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 100,
        vrs: ty.Optional[float] = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        tau_v: ty.Optional[float] = 0,
        tau_u: ty.Optional[float] = 0,
        dt: ty.Optional[float] = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )
        # Set threshold and reset voltage
        self.vth = Var(shape=(1,), init=vth)
        self.vrs = Var(shape=(1,), init=vrs)
        msg_var_par = f"Initialized attributes in process '{self.name}'
            
        # Print the values
        msg_var_par = f"""{msg_var_par}:
             u = {u}, v = {v}
             tau_u = {tau_u}, tau_v = {tau_v}
             du = {self.du.init}, dv = {self.dv.init}
             bias_mant = {self.bias_mant.init}, bias_exp = {self.bias_exp.init}
             vth = {self.vth.init}
             vrs = {self.vrs.init}
             dt = {dt}"""
        self.logger.debug(msg_var_par)
        
