import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.lax as lax

class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    bias: jax.Array
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey1, lkey2 = jrandom.split(key, 3)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.dropout = eqx.nn.Dropout(0.5)
        self.norm = eqx.nn.LayerNorm(hidden_size)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size//2, use_bias=True, key=lkey1)
        self.linear2 = eqx.nn.Linear(hidden_size//2, out_size, use_bias=False, key=lkey2)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input, key):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        
        key, subkey = jrandom.split(key)
        out = self.dropout(out, key=subkey)
        out = self.norm(out)
        out = self.linear1(out)
        out = self.linear2(out)
        
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(out + self.bias)
