import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx

from functools import partial
from jaxtyping import PRNGKeyArray, Array

from model import MLP, FNN, RNN
from util import establish_batches, extract_all_sections

from config import epochs, hidden_size, learning_rate, l2_eps, time_range, eps, eval_period


@eqx.filter_jit
def evaluate(model, x, y, key):
    inf_model = eqx.nn.inference_mode(model)
    subkeys = jrandom.split(key, x.shape[0])
    pred_y = jax.vmap(inf_model)(x, subkeys)
    n_correct = jnp.sum((pred_y > 0.5) == y)
    return (n_correct / x.shape[0])

@eqx.filter_jit
def evaluate_loss(model, x, y, key):
    inf_model = eqx.nn.inference_mode(model)
    subkeys = jrandom.split(key, x.shape[0])
    pred_y = jax.vmap(inf_model)(x, subkeys)
    cross_ent = -jnp.mean(y * jnp.log(pred_y + eps) + (1 - y) * jnp.log(1 - pred_y + eps))
    l2_reg = l2_eps * model.weight_sum()
    return cross_ent + l2_reg

@eqx.filter_value_and_grad
def compute_loss(model, x, y, key):
    subkeys = jrandom.split(key, x.shape[0])
    pred_y = jax.vmap(model)(x, subkeys)
    cross_ent = -jnp.mean(y * jnp.log(pred_y + eps) + (1 - y) * jnp.log(1 - pred_y + eps))
    l2_reg = l2_eps * model.weight_sum()
    return cross_ent + l2_reg


def main(
    key: PRNGKeyArray,
    train: dict,
    valid: dict,
    test: dict,
    channel_select: Array
) -> dict:
    train_perf = jnp.zeros((epochs,))
    train_loss = jnp.zeros((epochs,))
    valid_perf = jnp.zeros((epochs,))
    valid_loss = jnp.zeros((epochs,))
    test_pos_perf = jnp.zeros((epochs,))
    test_neg_perf = jnp.zeros((epochs,))

    valid["positive"] = valid["positive"][:,:,channel_select]
    valid["negative"] = valid["negative"][:,:,channel_select]
    train["positive"] = train["positive"][:,:,channel_select]
    train["negative"] = train["negative"][:,:,channel_select]
    
    test["positive"] = test["positive"][:,:,channel_select]
    test["negative"] = test["negative"][:,:,channel_select]
    
    test_pos_xs = test["positive"]
    test_neg_xs = test["negative"]
    test_pos_ys = jnp.ones((test_pos_xs.shape[0], 1))
    test_neg_ys = jnp.zeros((test_neg_xs.shape[0], 1))
    
    cached_extract = jax.jit(partial(extract_all_sections, n_sections=test['positive'].shape[1] - time_range + 1))
    test_pos_xs, test_pos_ys = cached_extract(test_pos_xs, test_pos_ys)
    
    test_neg_xs, test_neg_ys = cached_extract(test_neg_xs, test_neg_ys)
    
    # Shift the data by a random offset
    # key, subkey = jrandom.split(key)
    # offset = jrandom.randint(subkey, (test_pos_xs.shape[0],), 0, test_pos_xs.shape[1]-time_range+1)
    # test_pos_xs = jax.vmap(lambda idx: jnp.roll(test_pos_xs[idx], offset[idx], axis=0)[-time_range:])(jnp.arange(test_pos_xs.shape[0]))

    # key, subkey = jrandom.split(key)
    # offset = jrandom.randint(subkey, (test_neg_xs.shape[0],), 0, test_neg_xs.shape[1]-time_range+1)
    # test_neg_xs = jax.vmap(lambda idx: jnp.roll(test_neg_xs[idx], offset[idx], axis=0)[-time_range:])(jnp.arange(test_neg_xs.shape[0]))

    min_len = min(valid["positive"].shape[0], valid["negative"].shape[0])
    valid_xs = jnp.concatenate([valid["positive"][:min_len], valid["negative"][:min_len]], axis=0)
    valid_ys = jnp.concatenate([jnp.ones((min_len,1)), jnp.zeros((min_len,1))])

    valid_xs, valid_ys = cached_extract(valid_xs, valid_ys)

    # Shift the data by a random offset
    # key, subkey = jrandom.split(key)
    # offset = jrandom.randint(subkey, (valid_xs.shape[0],), 0, valid_xs.shape[1]-time_range+1)
    # valid_xs = jax.vmap(lambda idx: jnp.roll(valid_xs[idx], offset[idx], axis=0)[-time_range:])(jnp.arange(valid_xs.shape[0]))

    min_len = min(train["positive"].shape[0], train["negative"].shape[0])
    train_xs = jnp.concatenate([train["positive"][:min_len], train["negative"][:min_len]], axis=0)
    train_ys = jnp.concatenate([jnp.ones((min_len,1)), jnp.zeros((min_len,1))])

    train_xs, train_ys = cached_extract(train_xs, train_ys)

    # Shift the data by a random offset
    # key, subkey = jrandom.split(key)
    # offset = jrandom.randint(subkey, (train_xs.shape[0],), 0, train_xs.shape[1]-time_range+1)
    # train_xs = jax.vmap(lambda idx: jnp.roll(train_xs[idx], offset[idx], axis=0)[-time_range:])(jnp.arange(train_xs.shape[0]))


    key, model_key = jrandom.split(key)
    model = RNN(in_size=train_xs.shape[-1], out_size=1, hidden_size=hidden_size, key=model_key)
    # model = FNN(in_size=train_xs.shape[-1] * train_xs.shape[-2], out_size=1, hidden_size=hidden_size, key=model_key)
    # model = MLP(in_size=train_xs.shape[-1] * train_xs.shape[-2], out_size=1, hidden_size=hidden_size, depth=3, key=model_key)

    @eqx.filter_jit
    def make_step(key, model, opt_state, x, y):
        _, grads = compute_loss(model, x, y, key)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state
    
    cached_establish = jax.jit(partial(establish_batches, data=train, min_len=min_len))
    
    def single_epoch(key, model, opt_state):
        key, subkey = jrandom.split(key)
        batched_data = cached_establish(subkey)
        for (x, y) in zip(*batched_data):
            key, subkey = jrandom.split(key)
            model, opt_state = make_step(key, model, opt_state, x, y)

        return model, opt_state
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    for epoch in range(epochs):
        key, subkey = jrandom.split(key)
        model, opt_state = single_epoch(subkey, model, opt_state)

        key, subkey = jrandom.split(key)
        trkey, vakey, tepkey, tenkey  = jrandom.split(subkey, 4)
        acc = {'train': evaluate(model, train_xs, train_ys, trkey), 'valid': evaluate(model, valid_xs, valid_ys, vakey), 'test_pos': evaluate(model, test_pos_xs, test_pos_ys, tepkey), 'test_neg': evaluate(model, test_neg_xs, test_neg_ys, tenkey)}
        key, subkey = jrandom.split(key)
        loss = {'train': evaluate_loss(model, train_xs, train_ys, trkey), 'valid': evaluate_loss(model, valid_xs, valid_ys, vakey)}
        
        train_perf = train_perf.at[epoch].set(acc['train'])
        train_loss = train_loss.at[epoch].set(loss['train'])
        valid_perf = valid_perf.at[epoch].set(acc['valid'])
        valid_loss = valid_loss.at[epoch].set(loss['valid'])
        test_pos_perf = test_pos_perf.at[epoch].set(acc['test_pos'])
        test_neg_perf = test_neg_perf.at[epoch].set(acc['test_neg'])

        if (epoch+1) % eval_period == 0:
            jax.debug.print("Epoch: {}", epoch+1)
            # jax.debug.print("Train: {} Valid: {}", acc['train'], acc['valid'])

            # print(f"Epoch {epoch+1} Train: {loss['train']:.4f} Valid: {loss['valid']:.4f}", end="\r")

    results = {}
    
    results['Train'] = train_perf
    results['TrainLoss'] = train_loss
    results['Valid'] = valid_perf
    results['ValidLoss'] = valid_loss
    results['TestPos'] = test_pos_perf
    results['TestNeg'] = test_neg_perf
    
    return results