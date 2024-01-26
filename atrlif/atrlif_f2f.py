"""
ModelScaler is a basic container for the functionality required for model-wise 
f2f scaling in brian2lava. Each model in the library will have its own Scaler,
if this is not present then f2f scaling won't be supported for that model.
A dictionary containing lambda functions to perform the variable-wise scaling
for F2F conversion of the LIF model. A more meaningful table of the forward scalings:

    v       ->  A * v               = v' (same for all voltage related vars like theta and r)
    j       ->  A/alpha_t * j       = j'
    tau_v   ->  alpha_t * tau_v     = tau_v'
    tau_j   ->  alpha_t * tau_j     = tau_j'
    w       ->  A/alpha_t * w    = w'
    bias    ->  A/alpha_t * bias    = bias'
    dt      ->  alpha_t * dt        = dt'

    The minimum values for alpha_t and A are calculated by imposing all 
    values to be > 1:

    min_alpha_t = 1/dt
    min_A = max(1/v, alpha_t/j, alpha_t/w, alpha_t/b)

Note: All the lambda functions must have the same number of arguments (all the parameters required in the scaling)
"""


from brian2lava.utils.const import LOIHI2_SPECS
class ModelScaler:
    # This attribute is not used but can be useful for debugging
    process_class = 'ATRLIF'
    forward_ops = {
        'v' : lambda alpha_t,A: A,
        'dt': lambda alpha_t,A: alpha_t,
        'j': lambda alpha_t,A: A/alpha_t,
        'w': lambda alpha_t,A: A/alpha_t,
        'bias': lambda alpha_t,A: A/alpha_t,
        # Threshold scales as v
        'theta_0': lambda alpha_t,A: A,
        'theta': lambda alpha_t,A: A,
        'theta_step': lambda alpha_t,A: A,
        # Refractory variable scales also as v
        'r': lambda alpha_t,A: A,
    }
    # Variables to be MSB-aligned are defined in 'model.json'. To avoid copy-paste mistakes we 
    # define this variable at runtime using the instance from the F2F converter.
    msb_align_act = None 
    variables = {'v','j','theta','r'}
    const = None
    mant_exp = {'bias','w'}
    
    @staticmethod
    def max_val(var_name):
        if var_name in ModelScaler.variables:
            return LOIHI2_SPECS.Max_Variables
        elif var_name in ModelScaler.const:
            return LOIHI2_SPECS.Max_Constants
        else:
            return LOIHI2_SPECS.Max_Weights

    @staticmethod
    def min_scaling_params(variables):
        """
        Get the minimum scaling parameters to shift all of the parameters into 
        integer range. This is model specific.
        """
        # take the min of each variable
        dt,v,j,b = variables['dt'][0],variables['v'][0],variables['j'][0],variables['bias'][0]
        theta,r = variables['theta'][0], variables['r'][0]
        # If this neuron doesn't have any synapses connected to it, w won't be defined.
        w = variables['w'][0] if 'w' in variables else 0
        min_alpha_t = 1/dt
        # Avoid ZeroDivisionError
        params_to_max = []
        if v != 0:
            params_to_max.append(1/v)
        if theta != 0:
            params_to_max.append(1/theta)
        if r != 0:
            params_to_max.append(1/r)
        if j != 0:
            params_to_max.append(min_alpha_t/j)
        if w != 0:
            params_to_max.append(min_alpha_t/w)
        if b != 0:
            params_to_max.append(min_alpha_t/b)

        min_A = max(params_to_max)
        return {'alpha_t': min_alpha_t, 'A': min_A}
    
    @staticmethod
    def optimal_scaling_params(variables):
        """
        Since LIF neurons are static the optimal choice for the parameters
        corresponds to the maximal range of values allowed (so increasing 
        the scaling parameters to the largest possible)
        """
        return ModelScaler.max_scaling_params(variables)
    
    @staticmethod
    def max_scaling_params(variables):
        """
        The scaling of each variable shouldn't surpass the largest values
        allowed on Loihi2. This is not foolproof, but should be a good choice.
        In most cases we expect vth to be the one that defines the value of A.
        (The other parameters would have to be at least factor of 1/dt larger than vth)
        """
        from numpy import infty
        alpha_t = 1/variables['dt'][0]
        overall_max_A = infty
        max_A = infty
        for var_name, (var_min,var_max) in variables.items():
            # Avoid zero values
            if var_max == 0:
                continue
            max_val = ModelScaler.max_val(var_name) 
            
            # Account for the fact that some variables are represented with smaller bit-ranges.
            # Since we're interested in their true value after the alignment, we account for the implied shift
            # here.
            if var_name in ModelScaler.msb_align_act:
                max_val = max_val * 2**LOIHI2_SPECS.MSB_Alignment_Act
                
            if var_name == 'v':
                max_A = (max_val-1)/var_max
            # Here we have to account that the threshold voltage can increase
            # as a safety measure we allow 3 step sizes within the max range of values
            # TODO: This should be adapted based on user feedback.
            elif var_name == 'theta_0':
                step = variables['theta_step'][1]
                # Steps could also be negative? 
                var_max = max(var_max + 3*step, var_max)
                max_A = (max_val-1)/var_max
            elif var_name == 'r':
                theta = variables['theta_0'][1]
                step = variables['theta_step'][1]
                # Account for the same window as above but from the perspective of r
                var_max = max(var_max + theta + 3*step, var_max)
                max_A = (max_val-1)/var_max 
            elif var_name =='bias':
                max_A = (max_val-1)*alpha_t/var_max
            elif var_name == 'j':
                max_A = (max_val-1)*alpha_t/var_max

            # In this model it's important to allow spikes to
            # accumulate in the current, because we don't have
            # a real threshold!
            # MOST RELEVANT PARAMETER ===================
            elif var_name == 'w':
                var_max = 2*var_max
                max_A = (max_val-1)*alpha_t/var_max
            overall_max_A = min(max_A,overall_max_A)

        assert overall_max_A >= ModelScaler.min_scaling_params(variables)['A'], "Parameter ranges not compatible for F2F conversion."

        return {'alpha_t': alpha_t, 'A': overall_max_A}



    
