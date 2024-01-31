from functools import partial
from itertools import combinations

import numpy as np
from scipy.stats import unitary_group

import jax
from jax import numpy as jnp

import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype('complex64')


def HaarSampleGeneration(Ndata, n, seed):
    '''
    generate random haar states,
    used as inputs in the t=T step for backward denoise
    Args:
    Ndata: number of samples in dataset
    '''
    np.random.seed(seed)
    states_T = unitary_group.rvs(dim=2**n, size=Ndata)[:,:,0]

    return jnp.array(states_T)

def backCircuit(input, params, n_tot, L):
    '''
    the backward denoise parameteric quantum circuits,
    designed following the hardware-efficient ansatz
    output is the state before measurmeents on ancillas
    Args:
    input: input quantum state of n_tot qubits
    params: the parameters of the circuit
    n_tot: number of qubits in the circuits
    L: layers of circuit
    '''
    c = tc.Circuit(n_tot, inputs=input)

    for l in range(L):
        for i in range(n_tot):
            c.rx(i, theta=params[2* n_tot * l + i])
            c.ry(i, theta=params[2* n_tot* l + n_tot + i])

        for i in range(n_tot // 2):
            c.cz(2 * i, 2 * i + 1)

        for i in range((n_tot-1) // 2):
            c.cz(2 * i + 1, 2 * i + 2)

    return c.state()


def classifierCircuit(inputs, params, n, L):
    '''
    A variational classifier on input states
    Input:
    inputs: N quantum states in shape N*(2^n)
    params: parameters of variational classifier
    n: number of qubits
    L: number of layers
    Output:
    expectation of single Z-measurement on all states
    '''
    c = tc.Circuit(n, inputs=inputs)

    for l in range(L):
        for i in range(n):
            c.rx(i, theta=params[2* n * l + i])
            c.ry(i, theta=params[2* n* l + n + i])

        for i in range(n // 2):
            c.cz(2 * i, 2 * i + 1)

        for i in range((n-1) // 2):
            c.cz(2 * i + 1, 2 * i + 2)

    return c.expectation((tc.gates.z(), [0]))



class QGAN():
    def __init__(self, n, na, Lg, Lc):
        '''
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        L: layers of circuit in each backward step
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.Lg = Lg
        self.Lc = Lc
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=self.Lg), 
                                       vectorized_argnums=0))
        self.classCircuit_vmap = K.jit(K.vmap(partial(classifierCircuit, n=self.n, L=self.Lc), 
                                       vectorized_argnums=0))
        
    @partial(jax.jit, static_argnums=(0, ))
    def randomMeasure(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        n_batch = inputs.shape[0]
        m_probs = jnp.abs(jnp.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
        m_probs = jnp.log(jnp.sum(m_probs, axis=2))
        m_res = jax.random.categorical(jax.random.PRNGKey(42), m_probs)
        indices = 2 ** self.n * jnp.reshape(m_res, [-1, 1]) + jnp.arange(2 ** self.n)
        post_state = jnp.take_along_axis(inputs, indices, axis=1)
        post_state /= jnp.linalg.norm(post_state, axis=1)[:, jnp.newaxis]
        
        return post_state
    
    @partial(jax.jit, static_argnums=(0, ))
    def dataGenerate(self, inputs, params):
        '''
        Backward denoise process at step t: including PQC and random measurement
        Args:
        inputs: the input data set at step t
        params: the parameters for the PQC
        '''
        # outputs through quantum circuits before measurement
        zero_tensor = jnp.zeros(shape=(inputs.shape[0], 2**self.n_tot-2**self.n), dtype=jnp.complex64)
        inputs_full = jnp.concatenate([inputs, zero_tensor], axis=1)
        output_full = self.backCircuit_vmap(inputs_full, params)

        # perform measurement
        states_gen = self.randomMeasure(output_full)
        return states_gen
