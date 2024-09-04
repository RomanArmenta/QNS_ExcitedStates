import numpy as np
import jax.numpy as jnp
import jax
import math
from matplotlib import pyplot as plt
import netket as nk
from netket.operator.spin import sigmaz, sigmax
import json

from functools import partial

plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 300