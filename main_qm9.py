import argparse
import json
import torch
from tqdm import tqdm

from models.egnn_jax import get_edges_batch
from qm9.utils import calc_mean_mad
from utils.utils import get_model, get_loaders, set_seed
from flax.training import train_state
import jax.numpy as jnp
import jax
import optax
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def save_model(model, params, model_path, model_name):
    """
    Given a model, we save the parameters and hyperparameters.

    Inputs:
        model - Network object without parameters
        params - Parameters to save of the model
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = {'hidden_sizes': model.hidden_sizes,
                   'num_classes': model.num_classes}
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


def load_model(model_path, model_name, state=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        state - (Optional) If given, the parameters are loaded into this training state. Otherwise,
                a new one is created alongside a network architecture.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    # TODO check this in depth
    net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, 'rb') as f:
        params = pickle.load(f)
    state = state.replace(params=params)
    return state, net


def calculate_loss(params, apply_fn, batch):
    preds = apply_fn({'params': params}, *batch)
    loss = jnp.mean((preds - batch['labels']))  # TODO since l1 loss, otherwise **2 missing
    return loss


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    loss = calculate_loss(state.params, state.apply_fn, batch)
    return loss


def test_model(state, data_loader):
    """
    Test a model on a specified dataset.

    Inputs:
        state - Training state including parameters and model apply function.
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    true_preds, count = 0., 0
    for batch in data_loader:
        acc = eval_step(state, batch)
        batch_size = batch[0].shape[0]
        true_preds += acc * batch_size
        count += batch_size
    test_acc = true_preds / count
    return test_acc.item()


# TODO do this
def parse_batch(data):
    batch_size, n_nodes, _ = data['positions'].size()
    atom_positions = data['positions'].view(batch_size * n_nodes, -1)
    atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1)
    edge_mask = data['edge_mask']
    one_hot = data['one_hot']
    charges = data['charges']
    nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

    nodes = nodes.view(batch_size * n_nodes, -1)
    # nodes = torch.cat([one_hot, charges], dim=1)
    edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
    label = data[args.property]
    return data


def train_model(args, model_name, checkpoint_path):
    # # Generate model
    model = get_model(args)  # .to(args.device)

    # # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    mean, mad = calc_mean_mad(train_loader)

    for i in range(3):
        print(train_loader.dataset[i])
        #batch_size, n_nodes, _ = train_loader.dataset[i]['positions'].size()
        #print(batch_size)
        #print(args.batch_size)
        #print(n_nodes)

    # Dummy variables
    h = 5#jnp.ones((args.batch_size * n_nodes, n_feat))
    x = 5#jnp.ones((args.batch_size * n_nodes, x_dim))
    edges, edge_attr = 5,5#get_edges_batch(n_nodes, args.batch_size)

    # Get optimization objects
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), h, x, edges, edge_attr),
        tx=optax.adamw(args.lr, weight_decay=args.weight_decay)
    )

    # TODO do cosine annealing
    # lr_schedule = optax.cosine_decay_schedule(init_value=args.lr,
    #                                           decay_steps=args.epochs * steps_per_epoch,
    #                                           alpha=final_lr / init_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # lr = args.lr
    # weight_decay = args.weight_decay
    #
    # # Total number of steps (epochs * steps_per_epoch)
    # # You would need to know or define steps_per_epoch based on your dataset
    # total_steps = args.epochs * steps_per_epoch
    #
    # # Setup cosine decay for the learning rate without restarts
    # lr_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps, alpha=0)
    #
    # # Create the optimizer with weight decay
    # optimizer = optax.chain(
    #     optax.adam(learning_rate=lr_schedule, weight_decay=weight_decay)
    # )

    best_train_mae, best_val_mae, best_model = float('inf'), float('inf'), None
    train_scores = []
    val_scores = []

    for epoch in tqdm(range(args.epochs)):
        ############
        # Training #
        ############
        epoch_mae_train, epoch_mae_val = 0, 0
        for batch in tqdm(train_loader.dataset, desc=f"Epoch {epoch+1}", leave=False):
            state, mae = train_step(state, batch)
            epoch_mae_train += mae
        epoch_mae_train /= len(train_loader.dataset)
        train_scores.append(epoch_mae_train)

        ##############
        # Validation #
        ##############
        epoch_mae_val = test_model(state, val_loader.dataset)
        val_scores.append(epoch_mae_val)
        print(f"[Epoch {epoch + 1:2d}] Training accuracy: {epoch_mae_train:05.2%}, Validation accuracy: {epoch_mae_val:4.2%}")

        if len(val_scores) == 1 or epoch_mae_val > val_scores[best_val_epoch]:
            print("\t   (New best performance, saving model...)")
            save_model(model, state.params, checkpoint_path, model_name)
            best_val_epoch = epoch

    state, _ = load_model(checkpoint_path, model_name, state=state)
    test_mae = test_model(state, test_loader.dataset)
    results = {"test_mae": test_mae, "val_scores": val_scores,
               "train_scores": train_scores}
    with open(_get_result_file(checkpoint_path, model_name), "w") as f:
        json.dump(results, f)

    # Plot a curve of the validation accuracy
    sns.set()
    plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
    plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print((f" Test accuracy: {results['test_acc']:4.2%} ").center(50, "=") + "\n")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='egnn',
                        help='model')
    parser.add_argument('--num_hidden', type=int, default=77,
                        help='hidden features')
    parser.add_argument('--num_layers', type=int, default=7,
                        help='number of layers')
    parser.add_argument('--act_fn', type=str, default='silu',
                        help='activation function')


    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='learning rate')


    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--target_name', type=str, default='homo',
                        help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
    parser.add_argument('--dim', type=int, default=2,
                        help='ASC dimension')

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)
    train_model(parsed_args, 'test', 'assets')
