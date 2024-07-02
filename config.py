import jax.numpy as jnp
import jax.random as jrandom

k_folds: int = 5
epochs: int = 500
hidden_size: int = 6
learning_rate: float = 1e-3
batch_size: int = 16
eval_period: int = 20
shuffle: bool = False ### Shuffle labels
eps = 1e-5
l2_eps = 0#1e-2
time_range = 9
channel_selection = jnp.array([[0,1], [2,1]])

key = jrandom.PRNGKey(1) 