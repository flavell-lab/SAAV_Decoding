import jax
import jax.numpy as jnp
import jax.random as jrandom

from config import batch_size, time_range, shuffle

def rotate_data(arr, n_folds, fold_idx):
    indices = jnp.arange(arr.shape[0])
    perm = (indices - (arr.shape[0] // n_folds) * fold_idx) % arr.shape[0]
    return arr[perm], perm

def shuffle_data(arr, key):
    indices = jnp.arange(arr.shape[0])
    key, subkey = jrandom.split(key)
    perm = jrandom.permutation(subkey, indices)
    return arr[perm], perm


def extract_all_sections(data_xs, data_ys, n_sections):
    all_offsets = jnp.arange(0, n_sections)
    extracted_xs = jax.vmap(lambda xs: jax.vmap(lambda offset: jnp.roll(xs, offset, axis=0)[-time_range:])(all_offsets))(data_xs)
    extracted_xs = jnp.concatenate(extracted_xs, axis=0)
    
    extracted_ys = jnp.repeat(data_ys, n_sections, axis=0)
    
    return extracted_xs, extracted_ys

def establish_batches(key, data, min_len):
    # shuffle data and take the first min_len samples from each category
    key, subkey = jrandom.split(key)
    pos_cat = shuffle_data(data["positive"], subkey)[0][:min_len]
    key, subkey = jrandom.split(key)
    neg_cat = shuffle_data(data["negative"], subkey)[0][:min_len]
    
    # concatenate the two categories and shuffle them
    train_xs = jnp.concatenate([pos_cat, neg_cat], axis=0)
    train_ys = jnp.concatenate([jnp.ones((min_len,1)), jnp.zeros((min_len,1))], axis=0) 
    train_xs, train_ys = extract_all_sections(train_xs, train_ys, train_xs.shape[1] - time_range + 1)

    key, subkey = jrandom.split(key)
    train_xs, perm = shuffle_data(train_xs, subkey)
    
    # key, subkey = jrandom.split(key)
    train_ys = jax.lax.select(shuffle, train_ys, train_ys[perm]) # Shuffles labels with same permutation if shuffle is False

    # Shift the data by a random offset
    # key, subkey = jrandom.split(key)
    # offset = jrandom.randint(subkey, (train_xs.shape[0],), 0, train_xs.shape[1]-time_range+1)
    # train_xs = jax.vmap(lambda idx: jnp.roll(train_xs[idx], offset[idx], axis=0)[-time_range:])(jnp.arange(train_xs.shape[0]))

    print(f"Train_xs: {train_xs.shape}, Train_ys: {train_ys.shape}")
    truncated_len = train_xs.shape[0] - train_xs.shape[0]%batch_size
    print(f"Truncated len: {truncated_len}")
    
    
    # Batch the data
    batched = (train_xs[:truncated_len].reshape(-1, batch_size, train_xs.shape[1], train_xs.shape[2]), train_ys[:truncated_len].reshape(-1, batch_size, 1))
    print(f"Batch Size: {batch_size} n_batches: {batched[0].shape[0]}")
    
    return batched

