import jax.numpy as jnp
import jax.random as jrandom

k_folds: int = 5
# test_split = 1 / k_folds
epochs: int = 2000
hidden_size: int = 8
learning_rate: float = 1e-3
batch_size: int = 16
eval_period: int = 100
shuffle: bool = False ### Shuffle labels
eps = 1e-5
l2_eps = 0#1e-6
time_range = 3
channel_selection = jnp.array([[0,1], [2,1]])

key = jrandom.PRNGKey(1)