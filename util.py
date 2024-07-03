import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxtyping as jtyping

def rotate_data(arr, n_folds, fold_idx):
    indices = jnp.arange(arr.shape[0])
    perm = (indices - (arr.shape[0] // n_folds) * fold_idx) % arr.shape[0]
    return arr[perm], perm

def shuffle_data(key: jtyping.PRNGKeyArray, arr: jtyping.Array):
    indices = jnp.arange(arr.shape[0])
    key, subkey = jrandom.split(key)
    perm = jrandom.permutation(subkey, indices)
    return arr[perm], perm

def extract_all_sections(data_xs: jtyping.Array, data_ys: jtyping.Array, n_sections: int, hyperparams: dict):
    time_range = hyperparams["time_range"]
    all_offsets = jnp.arange(0, n_sections)
    extracted_xs = jax.vmap(lambda xs: jax.vmap(lambda offset: jnp.roll(xs, offset, axis=0)[-time_range:])(all_offsets))(data_xs)
    extracted_xs = jnp.concatenate(extracted_xs, axis=0)
    
    extracted_ys = jnp.repeat(data_ys, n_sections, axis=0)
    
    return extracted_xs, extracted_ys

def establish_batches(key: jtyping.PRNGKeyArray, data: dict, min_len: int, hyperparams: dict):
    """
    Establishes batches of training data for a given key, data, and minimum length.

    Args:
        key: A random key used for shuffling the data.
        data: A dictionary containing positive and negative samples.
        min_len: The minimum number of samples to take from each category.

    Returns:
        A tuple containing the batched training data and labels.
    """
    batch_size = hyperparams["batch_size"]
    time_range = hyperparams["time_range"]
    shuffle = hyperparams["shuffle"]

    # shuffle data and take the first min_len samples from each category
    key, subkey = jrandom.split(key)
    pos_cat = shuffle_data(subkey, data["positive"])[0][:min_len]
    key, subkey = jrandom.split(key)
    neg_cat = shuffle_data(subkey, data["negative"])[0][:min_len]
    
    # concatenate the two categories and shuffle them
    train_xs = jnp.concatenate([pos_cat, neg_cat], axis=0)
    train_ys = jnp.concatenate([jnp.ones((min_len,1)), jnp.zeros((min_len,1))], axis=0) 
    train_xs, train_ys = extract_all_sections(train_xs, train_ys, train_xs.shape[1] - time_range + 1, hyperparams)

    key, subkey = jrandom.split(key)
    train_xs, perm = shuffle_data(subkey, train_xs)
    
    train_ys = jax.lax.select(shuffle, train_ys, train_ys[perm]) # Shuffles labels with same permutation if shuffle is False

    print(f"Train_xs: {train_xs.shape}, Train_ys: {train_ys.shape}")
    truncated_len = train_xs.shape[0] - train_xs.shape[0]%batch_size
    print(f"Truncated len: {truncated_len}")
    
    # Batch the data
    batched = (train_xs[:truncated_len].reshape(-1, batch_size, train_xs.shape[1], train_xs.shape[2]), train_ys[:truncated_len].reshape(-1, batch_size, 1))
    print(f"Batch Size: {batch_size} n_batches: {batched[0].shape[0]}")
    
    return batched