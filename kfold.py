import jax
import jax.random as jrandom

from jaxtyping import PRNGKeyArray, Int

from train import main
from util import rotate_data

def single_fold(
    key: PRNGKeyArray, 
    i: Int,
    j: Int,
    data: Int,
    hyperparams: dict
):
    k_folds = hyperparams["k_folds"]

    fold_data = {}
    # Wrap around the data so each fold is different
    fold_data["positive"],_ = rotate_data(data["positive"], k_folds, i)
    fold_data["negative"],_ = rotate_data(data["negative"], k_folds, i)
    
    train_valid = {}
    # Sub-sample the training + validation data split
    train_valid["positive"] = rotate_data(fold_data["positive"][:int(data["positive"].shape[0]*(1-1/k_folds))], k_folds-1, j)[0]
    train_valid["negative"] = rotate_data(fold_data["negative"][:int(data["negative"].shape[0]*(1-1/k_folds))], k_folds-1, j)[0]

    test = {}
    # Define the test data split
    test["positive"] = fold_data["positive"][int(data["positive"].shape[0]*(1-1/k_folds)):]
    test["negative"] = fold_data["negative"][int(data["negative"].shape[0]*(1-1/k_folds)):]

    train = {}
    # Define the training data split
    train["positive"] = train_valid["positive"][:int(train_valid["positive"].shape[0]*(1-1/(k_folds-1)))]
    train["negative"] = train_valid["negative"][:int(train_valid["negative"].shape[0]*(1-1/(k_folds-1)))]

    valid = {}
    # Define the validation data split
    valid["positive"] = train_valid["positive"][int(train_valid["positive"].shape[0]*(1-1/(k_folds-1))):]
    valid["negative"] = train_valid["negative"][int(train_valid["negative"].shape[0]*(1-1/(k_folds-1))):]
    
    channel_selection = hyperparams["channel_selection"]

    keys = jrandom.split(key)
    results = jax.vmap(lambda ks, sel: main(ks, train, valid, test, sel, hyperparams))(keys, channel_selection)
        
    return results