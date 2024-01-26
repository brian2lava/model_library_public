import numpy as np
import typing as ty
from brian2.utils.logger import get_logger

from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort

class ProbSpiker(AbstractProcess):
    """Probabilistic spiker neuron. Spiking follows a Poisson process with a given
    probability. Does not use any synaptic input.

    Parameters
    ----------
    shape : tuple of int
        Shape of the group of neurons represented by this process.
    rnd : ndarray
        A multivariate random variable to determine the spiking of all neurons
        belonging to the group.
    p_spike : float, list of float, ndarray of float
        The probability that a spike occurs in the duration of a timestep. Can be
        specified individually for each neuron.
    name : str
        Name of the current process.
    dt : float
        Duration of one timestep.
    log_config : LogConfig
        Configuration options for logging.
    """
    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...] = (1,),
                 rnd: ty.Optional[np.ndarray] = 0,
                 p_spike: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
                 name: ty.Optional[str] = None,
                 dt: ty.Optional[float] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:
        super().__init__(shape=shape, 
                         name=name, 
                         log_config=log_config,
                         **kwargs)
        self.logger = get_logger('brian2.devices.lava')
        self.s_out = OutPort(shape=shape)
        self.rnd = Var(shape=shape, init=rnd)
        self.p_spike = Var(shape=shape, init=p_spike)

        # Make shape available in process model
        self.shape = Var(shape=(1,), init=shape)

        # Print the values
        msg_var_par = f"Initialized attributes in process '{self.name}'"
        msg_var_par = f"""{msg_var_par}:
                shape = {shape}
                rnd = {self.rnd.init}
                p_spike = {self.p_spike.init}
                dt = {dt}"""
        self.logger.debug(msg_var_par)