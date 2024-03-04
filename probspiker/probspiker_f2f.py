"""
ModelScaler is a basic container for the functionality required for model-wise 
f2f scaling in brian2lava. Each model in the library will have its own Scaler,
if this is not present then f2f scaling won't be supported for that model.
A dictionary containing lambda functions to perform the variable-wise scaling
for F2F conversion of the probspiker. Since this is only based on probabilities,
the F2F will just act as the identity operator. 
"""
from brian2lava.utils.const import LOIHI2_SPECS
class ModelScaler:
    process_class = 'LIF'
    forward_ops = {
        'p_spike' : lambda alpha_t,A: A,
        'rnd' : lambda alpha_t,A: A,
    }
    # It's useful to differentiate variables and constants
    # since they are treated differently by Loihi
    variables = {'rnd'}
    # Variables to be MSB-aligned are defined in 'model.json'. To avoid copy-paste mistakes we 
    # define this variable at runtime using the instance from the F2F converter.
    msb_align_act = None
    msb_align_decay = None
    msb_align_prob = None
    const = {'p_spike'}
    mant_exp = { }
    
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
        
        return {'alpha_t': 1, 'A': 1}
    
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
        Since the only parameters are probability-related,
        we don't need to convert anything, since probabilities are
        handled by brian2lava separately.
        """

        return {'alpha_t': 1, 'A': 1}



    
