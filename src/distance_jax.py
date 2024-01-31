import jax
from jax import numpy as jnp

from ott.geometry import pointcloud
from ott.solvers.linear import solve
from ott.geometry.costs import CostFn

from opt_einsum import contract
from functools import partial

import ot


@jax.jit
def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn',
                        jnp.conj(Set1), Set1)) ** 2.)
    r22 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn',
                        jnp.conj(Set2), Set2)) ** 2.)
    r12 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn',
                        jnp.conj(Set1), Set2)) ** 2.)

    return 2 * r12 - r11 - r22


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - jnp.abs(contract('mi,ni->mn', jnp.conj(Set1),
                     Set2, backend='jax')) ** 2.
    u0 = jnp.ones((D.shape[0],)) / D.shape[0]
    u1 = jnp.ones((D.shape[1],)) / D.shape[1]
    Wass_dis = ot.emd2(u0, u1, M=D)

    return Wass_dis


@jax.tree_util.register_pytree_node_class
class Trace(CostFn):
    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return 1. - jnp.abs(jnp.conj(x) @ y.T) ** 2.


@partial(jax.jit, static_argnums=(2, 3, 4, ))
def sinkhornDistance(Set1, Set2, reg=0.01, threshold=0.001, lse_mode=True):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    geom = pointcloud.PointCloud(Set1, Set2, cost_fn=Trace(), epsilon=reg)
    ot = solve(geom, a=None, b=None, lse_mode=lse_mode, threshold=threshold)

    return ot.reg_ot_cost
