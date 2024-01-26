import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ATRLIF(AbstractProcess):
	"""
	Adaptive Threshold Leaky-Integrate-and-Fire Process.
	With activation input and spike output ports a_in and s_out.

	# Init:
	j = 0
	v = 0
	theta = 0+1.5*5
	r = 0
	s = 0
	delta_j = (1<<12)*0.1
	delta_v = (1<<12)*0.1
	delta_theta = (1<<12)*0.1
	delta_r = (1<<12)*0.05
	theta_0 = 1.5*5
	theta_step = 1.5*5/2
	bias_mant = 0
	bias_exp = 0

	# Dynamics (see https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/slayer/neuron/alif.py,
	                https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/neuron_dynamics/dynamics.ipynb):
	j[t] = (1-delta_j)*j[t-1] + x[t]
    v[t] = (1-delta_v)*v[t-1] + j[t] + bias 
    theta[t] = (1-delta_theta)*(theta[t-1] - theta_0) + theta_0 
    r[t] = (1-delta_r)*r[t-1]
	
	# Spike event:
	s[t] = (v[t] - r[t]) >= theta[t]
	
	# Post spike event:
	r[t] = r[t] + 2*theta[t]
    theta[t] = theta[t] + theta_step

	Parameters
	----------
	shape : tuple(int)
		Number and topology of LIF neurons.
	j : float, list, numpy.ndarray, optional
		Initial value of the neuron's current.
	v : float, list, numpy.ndarray, optional
		Initial value of the neuron's voltage (membrane potential).
	theta : float, list, numpy.ndarray, optional
		Initial value of the threshold
	r : float, list, numpy.ndarray, optional
		Initial value of the refractory dynamics
	s : bool, list, numpy.ndarray, optional
		Initial spiking state
	delta_j : float, optional
		...
	delta_v : float, optional
		...
	delta_theta : float, optional
		...
	delta_r : float, optional
		...
	theta_0 : float, optional
		...
	theta_step : float, optional
		...
	bias_mant : float, list, numpy.ndarray, optional
		Mantissa part of neuron bias.
	bias_exp : float, list, numpy.ndarray, optional
		Exponent part of neuron bias, if needed. Mostly for fixed point
		implementations. Ignored for floating point implementations.

	Example
	-------
	>>> ad_th_lif = ATRLIF(shape=(200, 15), decay_theta=10, decay_v=5)
	This will create 200x15 ATRLIF neurons that all have the same current decay
	of 10 and voltage decay of 5.
	"""

	def __init__(
			self,
			*,
			shape: ty.Tuple[int, ...],
			j: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			theta: ty.Optional[ty.Union[float, list, np.ndarray]] = 5,
			r: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			s: ty.Optional[ty.Union[bool, list, np.ndarray]] = 0,
			delta_j: ty.Optional[float] = 0.4,
			delta_v: ty.Optional[float] = 0.4,
			delta_theta: ty.Optional[float] = 0.4,
			delta_r: ty.Optional[float] = 0.2,
			theta_0: ty.Optional[float] = 5,
			#theta_0: ty.Optional[ty.Union[float, list, np.ndarray]] = 5,
			theta_step: ty.Optional[float] = 3.75,
			bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			name: ty.Optional[str] = None,
			log_config: ty.Optional[LogConfig] = None,
            **kwargs) -> None:

		super().__init__(
			shape=shape,
			j=j,
			v=v,
			theta=theta,
			r=r,
			s=s,
			delta_j=delta_j,
			delta_v=delta_v,
			delta_theta=delta_theta,
			delta_r=delta_r,
			theta_0=theta_0,
			theta_step=theta_step,
			bias_mant=bias_mant,
			bias_exp=bias_exp,
			name=name,
			log_config=log_config,
            **kwargs)

		# Ports
		self.a_in = InPort(shape=shape)
		self.s_out = OutPort(shape=shape)

		# Bias
		self.bias_mant = Var(shape=shape, init=bias_mant)
		self.bias_exp = Var(shape=shape, init=bias_exp)
		#print(f"bias_mant = {bias_mant}, bias_exp = {bias_exp}")

		# Variables
		self.j = Var(shape=shape, init=j)
		self.v = Var(shape=shape, init=v)
		self.theta = Var(shape=shape, init=theta)
		self.r = Var(shape=shape, init=r)
		self.s = Var(shape=shape, init=s) # TODO used at all?

		# Parameters
		self.delta_j = Var(shape=(1,), init=delta_j)
		self.delta_v = Var(shape=(1,), init=delta_v)
		self.delta_theta = Var(shape=(1,), init=delta_theta)
		self.delta_r = Var(shape=(1,), init=delta_r)
		self.theta_0 = Var(shape=(1,), init=theta_0)
		self.theta_step = Var(shape=(1,), init=theta_step)

