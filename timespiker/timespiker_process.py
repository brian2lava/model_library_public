import numpy as np
import typing as ty
from brian2.utils.logger import get_logger

from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

class TimeSpiker(AbstractProcess):
    """Time-specific spiker neuron. Spiking occurs at given times. Accepts but does
    not use synaptic input.

    Parameters
    ----------
    shape : tuple of int
        Shape of the group of neurons represented by this process.
    t_rp_steps : int, optional
        The duration of the refractory period in timesteps.
    t_spike_steps : int, list of int, ndarray of int
        The timestep at which a spike shall occur. Can be specified individually
        for each neuron.
    name : str
        Name of the current process.
    log_config : LogConfig
        Configuration options for logging.
    """
    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...] = (1,),
                 t_rp_steps: ty.Optional[int] = 1,
                 t_rp_steps_end: ty.Optional[ty.Union[int, list, np.ndarray]] = -1,
                 t_spike_steps: ty.Optional[ty.Union[int, list, np.ndarray]] = 0,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:
        super().__init__(shape=shape, 
                         name=name, 
                         log_config=log_config,
                         **kwargs)
        self.logger = get_logger('brian2.devices.lava')
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.t_rp_steps = Var(shape=(1,), init=t_rp_steps)
        self.t_rp_steps_end = Var(shape=shape, init=t_rp_steps_end)
        self.t_spike_steps = Var(shape=shape, init=t_spike_steps)

        # Make shape available in process model
        self.shape = Var(shape=(1,), init=shape)

        # Print the values
        msg_var_par = f"Initialized attributes in process '{self.name}'"
        msg_var_par = f"""{msg_var_par}:
                shape = {shape}
                t_rp_steps = {self.t_rp_steps.init} (computed from t_rp)
                t_spike_steps = {self.t_spike_steps.init}"""
        self.logger.debug(msg_var_par)
        print(msg_var_par)