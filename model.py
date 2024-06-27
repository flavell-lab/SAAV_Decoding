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
    
    def weight_sum(self):
        return (jnp.sum((self.cell.weight_ih**2)) + jnp.sum((self.cell.weight_hh**2)) + jnp.sum((self.linear1.weight**2)) + jnp.sum((self.norm.weight**2)) + jnp.sum((self.linear2.weight**2)))


class FNN(eqx.Module):
    hidden_size: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    bias: jax.Array
    # norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, in_size, out_size, hidden_size, *, key):
        lkey1, lkey2 = jrandom.split(key, 2)
        self.hidden_size = hidden_size
        self.dropout = eqx.nn.Dropout(0.5)
        self.linear1 = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=lkey1)
        # self.norm = eqx.nn.LayerNorm(hidden_size)
        self.linear2 = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey2)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input, key):
        out = jnp.ravel(input)
        out = self.linear1(out)
        out = self.dropout(out, key=key)
        # out = self.norm(out)
        out = self.linear2(out)
        
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(out + self.bias)
    
    def weight_sum(self):
        return (jnp.sum((self.linear1.weight**2)) + jnp.sum((self.linear2.weight**2)))
        # return (jnp.sum((self.linear1.weight**2)) + jnp.sum((self.norm.weight**2)) + jnp.sum((self.linear2.weight**2)))



class MLP(eqx.Module):
    hidden_size: int
    first_layer: eqx.nn.Linear
    layers: list[eqx.nn.Linear]
    final_layer: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, depth, *, key):
        key, lkey1 = jrandom.split(key)
        lkeys = jrandom.split(key, depth) 
        self.hidden_size = hidden_size
        self.dropout = eqx.nn.Dropout(0.5)

        self.first_layer = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=lkey1)
        self.layers = [eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=k) for k in lkeys[:-1]]
        self.final_layer = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkeys[-1])

        self.bias = jnp.zeros(out_size)

    def __call__(self, input, key):
        out = jnp.ravel(input)

        out = self.first_layer(out)
        for layer in self.layers:
            out = layer(out)
            key, subkey = jrandom.split(key)
            out = self.dropout(out, key=subkey)
        
        out = self.final_layer(out)
        
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(out + self.bias)
    
    def weight_sum(self):
        return (jnp.sum((self.first_layer.weight**2)) + jnp.sum(jnp.array([jnp.sum((layer.weight**2)) for layer in self.layers])) + jnp.sum((self.final_layer.weight**2)))