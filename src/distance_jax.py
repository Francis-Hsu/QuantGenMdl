import jax
from jax import numpy as jnp

from opt_einsum import contract

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
    r11 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set1)) ** 2.)
    r22 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set2), Set2)) ** 2.)
    r12 = 1. - jnp.mean(jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set2)) ** 2.)
    
    return 2 * r12 - r11 - r22

def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set2, backend='jax')) ** 2.
    u0 = jnp.ones((D.shape[0],)) / D.shape[0]
    u1 = jnp.ones((D.shape[1],)) / D.shape[1]
    Wass_dis = ot.emd2(u0, u1, M=D)

    return Wass_dis

def sinkhornDistance(Set1, Set2, reg=0.005, eps=1e-4, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - jnp.abs(contract('mi,ni->mn', jnp.conj(Set1), Set2, backend='jax')) ** 2.
    u0 = jnp.ones((D.shape[0],)) / D.shape[0]
    u1 = jnp.ones((D.shape[1],)) / D.shape[1]
    if log == True:
        sh_dis = ot.sinkhorn2(u0, u1, M=D, reg=reg, stopThr=eps, method='sinkhorn_stabilized')
    else:
        sh_dis = ot.sinkhorn2(u0, u1, M=D, reg=reg, stopThr=eps)
        
    return sh_dis