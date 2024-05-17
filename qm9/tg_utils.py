from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import jax
import jraph
import jax.numpy as jnp
import jax.tree_util as tree
from typing import Callable, Dict, Tuple, NamedTuple

class NodeFeatures(NamedTuple):
    """Simple container for scalar and vectorial node features."""
    s: jnp.ndarray = None
    v: jnp.ndarray = None

def QM9GraphTransform(
    args,
    max_batch_nodes: int,
    max_batch_edges: int,
    train_trn: Callable,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jax GraphsTuple.
    """
    def _to_jax_graph(
        data: Dict, training: bool = True
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        senders = jnp.array(data['edge_index'][0], dtype=jnp.int32)
        receivers = jnp.array(data['edge_index'][1], dtype=jnp.int32)
        loc = jnp.array(data['pos'])

        nodes = NodeFeatures(
            s=jnp.array(data['z'], dtype=jnp.int32), v=None
        )

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=loc[senders] - loc[receivers],
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([data['num_nodes']]),
            n_edge=jnp.array([len(senders)]),
            globals=jnp.pad(loc, [(0, max_batch_nodes - loc.shape[0] - 1), (0, 0)]),
        )
        graph = jraph.pad_with_graphs(
            graph,
            n_node=max_batch_nodes,
            n_edge=max_batch_edges,
            n_graph=graph.n_node.shape[0] + 1,
        )

        target = jnp.array(data[args.prop])
        if args.task == 'node':
            target = jnp.pad(target, [(0, max_batch_nodes - target.shape[0] - 1)])
        if args.task == 'graph':
            target = jnp.append(target, 0)

        if training and train_trn is not None:
            target = train_trn(graph, target)

        return graph, target

    return _to_jax_graph

def pooling(
    graph: jraph.GraphsTuple,
    aggregate_fn: Callable = jraph.segment_sum,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pools over graph nodes with the specified aggregation."""
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    batch = jnp.repeat(graph_idx, graph.n_node, total_repeat_length=sum_n_node)

    s, v = None, None
    if graph.nodes.s is not None:
        s = aggregate_fn(graph.nodes.s, batch, n_graphs)
    if graph.nodes.v is not None:
        v = aggregate_fn(graph.nodes.v, batch, n_graphs)

    return s, v

def add_offsets(mean: float, atomrefs: jnp.ndarray) -> Callable:
    @jax.jit
    def _postprocess(graph: jraph.GraphsTuple, target: jnp.ndarray) -> jnp.ndarray:
        target = target + mean
        y0 = pooling(graph._replace(nodes=NodeFeatures(s=atomrefs[graph.nodes])))[0]
        target = (target + jnp.squeeze(y0)) * jraph.get_graph_padding_mask(graph)
        return target

    return _postprocess

def remove_offsets(mean: float, atomrefs: jnp.ndarray) -> Callable:
    @jax.jit
    def _postprocess(graph: jraph.GraphsTuple, target: jnp.ndarray) -> jnp.ndarray:
        target = target - mean * graph.n_node
        y0 = pooling(graph._replace(nodes=NodeFeatures(s=atomrefs[graph.nodes])))[0]
        target = (target - jnp.squeeze(y0)) * jraph.get_graph_padding_mask(graph)
        return target

    return _postprocess

def setup_qm9_data(
    args,
) -> Tuple[DataLoader, DataLoader, DataLoader, Callable, Callable]:
    # we are not doing it TODO check
    # transforms = [
    #     T.SubtractCenterOfMass(),
    #     T.MatScipyNeighborList(args.radius),
    #     T.CastTo32(),
    # ]
    dataset = QM9(root='./dataset/QM9', pre_transform=T.Distance())
    
    mean = float(dataset.data.y.mean())
    atomref = jnp.array(
        dataset.data[0].atomref, dtype=jnp.float32
    )
    train_target_transform = remove_offsets(mean, atomref)
    eval_target_transform = add_offsets(mean, atomref)

    max_batch_nodes = int(
        1.3 * max([data.num_nodes for data in dataset])
    )
    max_batch_edges = int(1.3 * max([data.edge_index.size(1) for data in dataset]))

    to_graphs_tuple = QM9GraphTransform(
        args,
        max_batch_nodes=max_batch_nodes,
        max_batch_edges=max_batch_edges,
        train_trn=train_target_transform,
    )

    loader_train = DataLoader(dataset[:100000], batch_size=32, shuffle=True)
    loader_val = DataLoader(dataset[100000:110000], batch_size=32)
    loader_test = DataLoader(dataset[110000:], batch_size=32)

    return (
        loader_train,
        loader_val,
        loader_test,
        to_graphs_tuple,
        eval_target_transform,
    )