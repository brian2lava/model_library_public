import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

@implements(proc=ATRLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyATRLIFModelFloat(PyLoihiProcessModel):
	"""
	Implementation of Adaptive-Threshold Leaky-Integrate-and-Fire neuron process in floating-point precision.
	This short and simple ProcessModel can be used for quick algorithmic prototyping, without engaging with the 
	nuances of a fixed-point implementation.
	"""
	a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
	s_out = None
	j: np.ndarray = LavaPyType(np.ndarray, float)
	v: np.ndarray = LavaPyType(np.ndarray, float)
	theta: np.ndarray = LavaPyType(np.ndarray, float)
	r: np.ndarray = LavaPyType(np.ndarray, float)
	s: np.ndarray = LavaPyType(np.ndarray, bool)
	bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
	bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
	delta_j: float = LavaPyType(float, float)
	delta_v: float = LavaPyType(float, float)
	delta_theta: float = LavaPyType(float, float)
	delta_r: float = LavaPyType(float, float)
	theta_0: float = LavaPyType(float, float)
	theta_step: float = LavaPyType(float, float)
	s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
	

	def __init__(self, proc_params):
		super(PyATRLIFModelFloat, self).__init__(proc_params)
		print("PyATRLIFModelFloat initialized")


	def subthr_dynamics(self, activation_in: np.ndarray):
		"""
		Common sub-threshold dynamics for Adaptive-Threshold LIF models.
		j[t] = (1-delta_j)*j[t-1] + x[t] 
		v[t] = (1-delta_v)*v[t-1] + j[t] + bias_mant
		theta[t] = (1-delta_theta)*(theta[t-1] - theta_0) + theta_0 
		r[t] = (1-delta_r)*r[t-1]
		"""
		self.j[:] = (1-self.delta_j)*self.j + activation_in
		self.v[:] = (1-self.delta_v)*self.v + self.j + self.bias_mant
		self.theta[:] = (1-self.delta_theta)*(self.theta - self.theta_0) + self.theta_0 
		self.r[:] = (1-self.delta_r)*self.r
		#print(f"-  delta_j = {self.delta_j}\n-  j_updated = {self.j}")
		#print(f"-  delta_v = {self.delta_v}\n-  v_updated = {self.v}")

	
	def post_spike(self, spike_vector: np.ndarray):
		"""
		Post spike/refractory behavior:
		r[t] = r[t] + 2*theta[t]
    	theta[t] = theta[t] + theta_step
		"""
		r_spiking = self.r[spike_vector]
		theta_spiking = self.theta[spike_vector]

		self.r[spike_vector] = r_spiking + 2*theta_spiking
		self.theta[spike_vector] = theta_spiking + self.theta_step


	def run_spk(self):
		"""
		The run function processing a spike event for Adaptive-Threshold LIF, which
		occurs if (v[t] - r[t]) >= theta[t].
		"""
		#super().run_spk()
		a_in_data = self.a_in.recv()

		self.subthr_dynamics(activation_in=a_in_data)
		self.s[:] = (self.v - self.r) >= self.theta
		self.s_out_buff = self.s
		#print(f"s_out_buff = {self.s_out_buff}")
		self.post_spike(spike_vector=self.s_out_buff)
		self.s_out.send(self.s_out_buff)


@implements(proc=ATRLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyATRLIFModelFixed(PyLoihiProcessModel):
	"""
	Implementation of Adaptive-Threshold Leaky-Integrate-and-Fire neuron process in fixed-point precision,
	bit-by-bit mimicking the fixed-point computation behavior of Loihi 2.
	"""
	a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
	j: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
	v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
	theta: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
	r: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
	s: np.ndarray = LavaPyType(np.ndarray, bool)
	bias_mant: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
	bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)
	delta_j: int = LavaPyType(int, np.uint16, precision=12)
	delta_v: int = LavaPyType(int, np.uint16, precision=12)
	delta_theta: int = LavaPyType(int, np.uint16, precision=12)
	delta_r: int = LavaPyType(int, np.uint16, precision=12)
	theta_0: int = LavaPyType(int, np.uint16, precision=12)
	theta_step: int = LavaPyType(int, np.uint16, precision=12)
	s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)


	def __init__(self, proc_params):
		super(PyATRLIFModelFixed, self).__init__(proc_params)
		print("PyATRLIFModelFixed initialized")
		
		# The `ds_offset` constant enables setting decay constant values to exact 4096 = 2**12. 
		# Without it, the range of 12-bit unsigned decay_u and dv is 0 to 4095.
		self.ds_offset = 1
		self.isthrscaled = False
		self.effective_bias = 0
		# State variables j and v are 24 bits wide
		self.jv_bitwidth = 24
		self.max_jv_val = 2 ** (self.jv_bitwidth - 1)
		# Decays need an MSB alignment by 12 bits
		# --> already done by Brian2Lava!
		self.decay_shift = 12
		self.decay_unity = 2**self.decay_shift
		# Threshold is MSB-aligned by 6 bits
		self.theta_unity = 2**6
        # Incoming activation is MSB-aligned by 6 bits
		self.act_unity = 2**6

	
	def subthr_dynamics(self, activation_in: np.ndarray):
		"""
		Common sub-threshold dynamics for adaptive-threshold LIF models.
		j[t] = (1-delta_j)*j[t-1] + x[t]
		v[t] = (1-delta_v)*v[t-1] + j[t] + bias_mant 
		theta[t] = (1-delta_theta)*(theta[t-1] - theta_0) + theta_0 
		r[t] = (1-delta_r)*r[t-1]
		"""
		# Update current
		# --------------
		# Compute decay constant (left shift via multiplication by `decay_unity`
        # --> already done by Brian2Lava!)
		#decay_const_j = self.delta_j*self.decay_unity + self.ds_offset
		decay_const_j = self.delta_j + self.ds_offset

		# Below, j is promoted to int64 to avoid overflow of the product
		# between j and decay constant beyond int32. Subsequent right shift by
		# 12 brings us back within 24-bits (and hence, within 32-bits)
		j_decayed = np.int64(self.j * (self.decay_unity - decay_const_j)) # multiplication by 'decay_unity' is like a left shift
		j_decayed = np.sign(j_decayed) * np.right_shift(
			np.abs(j_decayed), self.decay_shift
		)
		# Hardware left-shifts synaptic input for MSB alignment
		activation_in = activation_in * self.act_unity # multiplication by 'act_unity' is like a left shift (cf. line below)
		#activation_in = np.left_shift(activation_in, act_shift)

		# Add synaptic input to decayed current
		j_updated = np.int32(j_decayed + activation_in)
		# Check if value of current is within bounds of 24-bit. Overflows are
		# handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
		# becomes k and -(2**23 + k) becomes -k
		wrapped_curr = np.where(
			j_updated > self.max_jv_val,
			j_updated - 2 * self.max_jv_val,
			j_updated,
		)
		wrapped_curr = np.where(
			wrapped_curr <= -self.max_jv_val,
			j_updated + 2 * self.max_jv_val,
			wrapped_curr,
		)
		#print(f"-  delta_j = {self.delta_j}\n-  decay_const_j = {decay_const_j}\n-  j_decayed = {j_decayed}\n-  j_updated = {wrapped_curr}")
		self.j[:] = wrapped_curr

		# Update voltage (decaying similar to current)
		# --------------------------------------------
		# Compute decay constant (left shift via multiplication by `decay_unity`
        # --> already done by Brian2Lava!)
		#decay_const_v = self.delta_v*self.decay_unity
		decay_const_v = self.delta_v

		neg_voltage_limit = -np.int32(self.max_jv_val) + 1
		pos_voltage_limit = np.int32(self.max_jv_val) - 1
		v_decayed = np.int64(self.v) * np.int64(self.decay_unity - decay_const_v) # multiplication by 'decay_unity' is like a left shift
		v_decayed = np.sign(v_decayed) * np.right_shift(
			np.abs(v_decayed), self.decay_shift
		)

		v_updated = np.int32(v_decayed + self.j + self.effective_bias)
		#print(f"-  delta_v = {self.delta_v}\n-  decay_const_v = {decay_const_v}\n-  v_decayed = {v_decayed}\n-  v_updated = {v_updated}")
		self.v[:] = np.clip(v_updated, neg_voltage_limit, pos_voltage_limit)

		# Update threshold (decaying similar to current)
		# ----------------------------------------------
		# Compute decay constant (left shift via multiplication by `decay_unity`
        # --> already done by Brian2Lava!)
		#decay_const_theta = self.delta_theta*self.decay_unity
		decay_const_theta = self.delta_theta

		theta_diff_decayed = np.int64(self.theta - self.theta_0) * \
		                     np.int64(self.decay_unity - decay_const_theta) # multiplication by 'decay_unity' is like a left shift
		theta_diff_decayed = np.sign(theta_diff_decayed) * np.right_shift(
			np.abs(theta_diff_decayed), self.decay_shift
		)
		#print(f"-  delta_theta = {self.delta_theta}\n-  decay_const_theta = {decay_const_theta}\n-  theta = {self.theta}\n-  theta_diff_decayed = {theta_diff_decayed}")

		self.theta[:] = np.int32(theta_diff_decayed) + self.theta_0
		# TODO clipping?

		# Update refractoriness (decaying similar to current)
		# ---------------------------------------------------
		# Compute decay constant (left shift via multiplication by `decay_unity`
        # --> already done by Brian2Lava!)
		#decay_const_r = self.delta_r*self.decay_unity
		decay_const_r = self.delta_r

		r_decayed = np.int64(self.r) * np.int64(self.decay_unity - decay_const_r) # multiplication by 'decay_unity' is like a left shift
		r_decayed = np.sign(r_decayed) * np.right_shift(
			np.abs(r_decayed), self.decay_shift
		)

		self.r[:] = np.int32(r_decayed)
		# TODO clipping?


	def scale_bias(self):
		"""
		Scale bias with bias exponent by taking into account sign of the exponent.
		"""
		# Create local copy of bias_mant with promoted dtype to prevent
		# overflow when applying shift of bias_exp.
		bias_mant = self.bias_mant.copy().astype(np.int32)
		self.effective_bias = np.where(
			self.bias_exp >= 0,
			np.left_shift(bias_mant, self.bias_exp),
			np.right_shift(bias_mant, -self.bias_exp),
		)
		#print(f"scale_bias():\n\tbias_mant = {self.bias_mant}\n\tbias_exp = {self.bias_exp}\n\teffective_bias = {self.effective_bias}")

	
	def scale_threshold(self):
		"""
		Scale threshold according to the way Loihi hardware scales it. In Loihi hardware,
		threshold is left-shifted by 6-bits to MSB-align it with other state variables of higher precision.
		"""
		self.theta_0 = np.int32(self.theta_0 * self.theta_unity) # multiplication equaling left shift
		self.theta = np.full(self.theta.shape, self.theta_0)
		self.theta_step = np.int32(self.theta_step * self.theta_unity) # multiplication equaling left shift
		self.isthrscaled = True

	
	def post_spike(self, spike_vector: np.ndarray):
		"""
		Post spike/refractory behavior:
		r[t] = r[t] + 2*theta[t]
    	theta[t] = theta[t] + theta_step
		"""
		r_spiking = self.r[spike_vector]
		theta_spiking = self.theta[spike_vector]

		self.r[spike_vector] = r_spiking + 2*theta_spiking
		self.theta[spike_vector] = theta_spiking + self.theta_step


	def run_spk(self):
		"""The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol. Processes spike events that occur if 
		(v[t] - r[t]) >= theta[t].
		"""
		#super().run_spk()

		# Receive synaptic input
		a_in_data = self.a_in.recv()

		# Compute effective bias
		self.scale_bias()

		# Compute scaled threshold-related variables only once, not every time-step
		# (has to be done once after object construction)
		if not self.isthrscaled:
			self.scale_threshold()		

		self.subthr_dynamics(activation_in=a_in_data)
		self.s[:] = (self.v - self.r) >= self.theta
		self.s_out_buff = self.s
		self.post_spike(spike_vector=self.s_out_buff)
		self.s_out.send(self.s_out_buff)
