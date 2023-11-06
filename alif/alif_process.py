import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ALIF(AbstractProcess):
	"""
	Adaptive Leaky-Integrate-and-Fire (aLIF) neural Process.
	With activation input and spike output ports a_in and s_out.

	LIF dynamics abstracts to:
	I[t] = I[t-1] * (1-dI) + a_in		 # neuron current
	u[t] = u[t-1] * (1-du) + I[t] + bias  # neuron voltage
	s_out = u[t] > vth					# spike if threshold is exceeded
	u[t] = 0							  # reset at spike

	ALIF dynamics abstracts to:
	# Init
	w = 0
	I = 0
	u = 0
	u_rest = 0
	decay_I = 0  # decay I
	decay_u = 0  # decay u
	decay_w = 0  # decay w
	beta = 0  # b
	delta_w = 0  # delta_w
	vth = 10
	bias_mant = 0
	bias_exp = 0
	y = u - u_rest
	# Dynamics
	I[t] = I[t-1] * (1-decay_I) + a_in
	y[t] = y[t-1] * (1-decay_u) - w[t-1] + I[t-1] + bias
	w[t] = w[t-1] * (1-decay_w) + y[t-1] * beta
	# Spike
	s_out = y[t] >= vth
	# Reset
	y[t] = 0
	w[t] = w[t-1] + delta_w

	Parameters
	----------
	shape : tuple(int)
		Number and topology of LIF neurons.
	I : float, list, numpy.ndarray, optional
		Initial value of the neurons' current.
	u : float, list, numpy.ndarray, optional
		Initial value of the neurons' voltage (membrane potential).
	w : float, list, numpy.ndarray, optional
		Initial value
	decay_I : float, optional
		Inverse of decay time-constant for current decay. Currently, only a
		single decay can be set for the entire population of neurons.
	decay_u : float, optional
		Inverse of decay time-constant for voltage decay. Currently, only a
		single decay can be set for the entire population of neurons.
	decay_w : float, optional
		Inverse of decay time-constant
	beta : float, optional
		Inverse of decay time-constant
	delta_w : float, optional
		Decay time-constant
	bias_mant : float, list, numpy.ndarray, optional
		Mantissa part of neuron bias.
	bias_exp : float, list, numpy.ndarray, optional
		Exponent part of neuron bias, if needed. Mostly for fixed point
		implementations. Ignored for floating point implementations.
	vth : float, optional
		Neuron threshold voltage, exceedecay_Ing which, the neuron will spike.
		Currently, only a single threshold can be set for the entire
		population of neurons.

	Example
	-------
	>>> alif = ALIF(shape=(200, 15), decay_I=10, decay_u=5)
	This will create 200x15 ALIF neurons that all have the same current decay
	of 10 and voltage decay of 5.
	"""

	def __init__(
			self,
			*,
			shape: ty.Tuple[int, ...],
			I: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			w: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			decay_I: ty.Optional[float] = 0,
			decay_u: ty.Optional[float] = 0,
			decay_w: ty.Optional[float] = 0,
			beta: ty.Optional[float] = 0,
			delta_w: ty.Optional[float] = 0,
			bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
			vth: ty.Optional[float] = 10,
			u_rest: ty.Optional[float] = 0,
			name: ty.Optional[str] = None,
			log_config: ty.Optional[LogConfig] = None
		) -> None:

		super().__init__(
			shape=shape,
			I=I,
			u=u,
			w=w,
			decay_I=decay_I,
			decay_u=decay_u,
			decay_w=decay_w,
			beta=beta,
			delta_w=delta_w,
			bias_mant=bias_mant,
			bias_exp=bias_exp,
			name=name,
			log_config=log_config
		)

		# Ports
		self.a_in = InPort(shape=shape)
		self.s_out = OutPort(shape=shape)

		# Bias
		self.bias_exp = Var(shape=shape, init=bias_exp)
		self.bias_mant = Var(shape=shape, init=bias_mant)

		# Threshold
		self.vth = Var(shape=(1,), init=vth)

		# Reset
		self.u_rest = Var(shape=(1,), init=u_rest)

		# Variables
		self.I = Var(shape=shape, init=I)
		self.u = Var(shape=shape, init=u)
		self.w = Var(shape=shape, init=w)
		self.y = Var(shape=shape, init=u-u_rest)  # Derived

		# Parameters
		self.decay_I = Var(shape=(1,), init=decay_I)
		self.decay_u = Var(shape=(1,), init=decay_u)
		self.decay_w = Var(shape=(1,), init=decay_w)
		self.beta = Var(shape=(1,), init=beta)
		self.delta_w = Var(shape=(1,), init=delta_w)
