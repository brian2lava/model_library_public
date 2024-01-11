"""
ModelScaler is a basic container for the functionality required for model-wise 
f2f scaling in brian2lava. Each model in the library will have its own Scaler,
if this is not present then f2f scaling won't be supported for that model.
A dictionary containing lambda functions to perform the variable-wise scaling
for F2F conversion of the LIF model. A more meaningful table of the forward scalings:

    v       ->  A * v               = v'
    j       ->  A/alpha_t * j       = j'
    tau_v   ->  alpha_t * tau_v     = tau_v'
    tau_j   ->  alpha_t * tau_j     = tau_j'
    w       ->  A/alpha_t**2 * w    = w'
    bias    ->  A/alpha_t * bias    = bias'
    dt      ->  alpha_t * dt        = dt'

    The minimum values for alpha_t and A are calculated by imposing all 
    values to be > 1:

    min_alpha_t = 1/dt
    min_A = max(1/v, alpha_t/j, alpha_t**2/w, alpha_t/b)

Note: All the lambda functions must have the same number of arguments (all the parameters required in the scaling)
"""
from brian2lava.utils.const import LOIHI2_SPECS
class ModelScaler:
    process_class = 'LIF'
    forward_ops = {
        'v' : lambda alpha_t,A: A,
        'v_rs': lambda alpha_t,A: A,
        'v_th': lambda alpha_t,A: A,
        'dt': lambda alpha_t,A: alpha_t,
        'j': lambda alpha_t,A: A/alpha_t,
        'tau_v': lambda alpha_t,A: alpha_t,
        'tau_j': lambda alpha_t,A: alpha_t,
        'w': lambda alpha_t,A: A/alpha_t**2,
        'bias': lambda alpha_t,A: A/alpha_t,
    }

    @staticmethod
    def min_scaling_params(variables):
        """
        Get the minimum scaling parameters to shift all of the parameters into 
        integer range. This is model specific.
        """
        # take the min of each variable
        dt,v,j,w,b = variables['dt'][0],variables['v'][0],variables['j'][0],variables['w'][0],variables['bias'][0]
        min_alpha_t = 1/dt
        # Avoid ZeroDivisionError
        params_to_max = []
        if v != 0:
            params_to_max.append(1/v)
        if j != 0:
            params_to_max.append(min_alpha_t/j)
        if w != 0:
            params_to_max.append(min_alpha_t**2/w)
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
    def max_scaling_params(variables, loihi_specs):
        """
        The scaling of each variable shouldn't surpass the largest values
        allowed on Loihi2. This is not foolproof, but should be a good choice.
        In most cases we expect vth to be the one that defines the value of A.
        (The other parameters would have to be at least factor of 1/dt larger than vth)
        """
        from numpy import infty
        alpha_t = 1/variables['dt'][0]
        overall_max_A = infty
        for varname, (var_min,var_max) in variables.values():
            max_val = LOIHI2_SPECS.Max_Voltage if varname!= 'w' else LOIHI2_SPECS.Max_Weights
            if varname in ['v','v_th','v_rs']:
                max_A = (max_val-1)/var_max
            elif varname in ['j','bias']:
                max_A = (max_val-1)*alpha_t/var_max
            elif varname == 'w':
                max_A = (max_val-1)*alpha_t**2/var_max
            overall_max_A = min(max_A,overall_max_A)
        
        assert overall_max_A >= ModelScaler.min_scaling_params(**variables)['A'], "Parameter ranges not compatible for F2F conversion."

        return {'alpha_t': alpha_t, 'A': overall_max_A}



    