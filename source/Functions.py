from typing import Union, Any
from flax import linen as nn
from jax import numpy as jnp
import numpy as np

# Restricted Boltzmann Machine class
class RBM(nn.Module):
    alpha: Union[float, int] = 1
    # def __init__(self, n_input: int, alpha = 1):

    #     assert alpha > 0, f'Parameter alpha is not greater than zero!!'
    #     assert n_input > 1, f'Parameter n_input is not greater than one!!'

    #     # Assign to self object
    #     self.alpha = alpha
    #     self.n_input = n_input

    # This function is called when call as RestrictedBoltzmannMachine(v), where v
    # is an input vector. Returns the marginal probability of the input vector v.
    # This function must give the log probability of the RBM, given by:
    # dot(a, v) + sum_i(log(2cosh(b_j + dot(W_i, v))))
    @nn.compact
    def __call__(self, v):

        # Initialize the layer. This gives the b_j's (hidden bias) and
        # the W_ij (the weights), then apply to the input v to return
        # a vector of n_hidden with entries b_j + dot(W_j, v).
        x = nn.Dense(
            name = "Dense",
            features = int(self.alpha * v.shape[-1]),
            param_dtype = np.float64,
            use_bias = True,
            kernel_init = nn.initializers.normal(),
            bias_init = nn.initializers.normal(),
        )(v)
        
        # Apply the non-lineal activation function log(cosh) to the whole vector
        x = jnp.log(jnp.cosh(x))

        # Sum over all activated terms
        x = jnp.sum(x, axis = -1)

        # Initialize the biases for the input layer.
        v_bias = self.param(
            "visible_bias",
            nn.initializers.normal(),
            (v.shape[-1],),
            np.float64,
        )

        # This gives the term sum(a_i v_i) where a_i are the v_biases
        out_bias = jnp.dot(v, v_bias)

        # Sum both and return it.
        return x + out_bias