import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from models.alif.alif_process import Process


@implements(proc=Process, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyALifModelFloat(AbstractPyALifModelFloat):
	"""
	Implementation of Leaky-Integrate-and-Fire neural process in **floating point** precision.
	This ProcessModel can be used for quick algorithmic prototyping.
	"""

	a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
	s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
	I: np.ndarray = LavaPyType(np.ndarray, float)
	u: np.ndarray = LavaPyType(np.ndarray, float)
	y: np.ndarray = LavaPyType(np.ndarray, float)
	w: np.ndarray = LavaPyType(np.ndarray, float)
	bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
	bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
	decay_I: float = LavaPyType(float, float)
	decay_u: float = LavaPyType(float, float)
	decay_w: float = LavaPyType(float, float)
	beta: float = LavaPyType(float, float)
	delta_w: float = LavaPyType(float, float)
	vth: float = LavaPyType(float, float)
	u_rest: float = LavaPyType(float, float)


	def __init__(self, proc_params):
		super(PyALifModelFloat, self).__init__(proc_params)
		
		print("PyLifModelFloat initialized")

		# TODO update variables
		# * remove bias
		# * remove dI
		# * replace decay_* by inverse: 1 - decay_*
		# * pass y directly (instead of u & u_rest)
		
		# TODO write wrapper function that creates process & process model based on original variables


	def spike_condition(self):
		"""
		Spiking activation function for adaptive LIF.
		"""

		return self.y >= self.vth


	def subthr_dynamics(self, activation_in: np.ndarray):
		"""
		Common sub-threshold dynamics of current and voltage variables for all adaptive LIF models.

		ALIF dynamics:
		I[t] = I[t-1] * (1-decay_I) + a_in
		y[t] = y[t-1] * (1-decay_u) - w[t-1] + I[t-1] + bias
		w[t] = w[t-1] * (1-decay_w) + y[t-1] * beta

		# Spike
		s_out = y[t] >= vth

		# Reset
		y[t] = 0
		w[t] = w[t-1] + delta_w

		u[t] = u[t-1] * (1-decay_u) + I[t] + bias
		y[t] = a*y[t-1] - w[t-1] + I[t-1] + bias
		"""

		self.I[:] += activation_in
		self.y[:] = self.y * (1 - self.decay_u) - self.w + self.I + self.bias_mant
		self.w[:] = self.w * (1 - self.decay_w) + self.y * self.beta


	def reset_voltage(self, spike_vector: np.ndarray):
		"""
		Voltage reset behaviour.
		"""

		self.y[spike_vector] = 0
		self.w[spike_vector] = self.w[spike_vector] + self.delta_w


	def run_spk(self):
		"""
		The run function that performs the actual computation during execution
		orchestrated by a PyLoihiProcessModel using the LoihiProtocol.
		"""

		#super().run_spk()
		a_in_data = self.a_in.recv()

		self.subthr_dynamics(activation_in=a_in_data)
		self.s_out_buff = self.spike_condition()
		self.reset_voltage(spike_vector=self.s_out_buff)
		self.s_out.send(self.s_out_buff)
