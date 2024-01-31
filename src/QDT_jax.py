import numpy as np
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group

import jax
from jax import numpy as jnp

from opt_einsum import contract
from functools import partial
from itertools import combinations

K = tc.set_backend("jax")
tc.set_dtype('complex64')

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
            c.rx(i, theta=params[2*n_tot*l+i])
            c.ry(i, theta=params[2*n_tot*l+n_tot+i])
        for i in range(n_tot//2):
            c.cz(2*i, 2*i+1)
        for i in range((n_tot-1)//2):
            c.cz(2*i+1, 2*i+2)
    return c.state()

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

class QDT():
    def __init__(self, n, na, L):
        '''
        the QDT model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        L: layers of circuit
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), 
                                             vectorized_argnums=0))

    def set_diffusionSet(self, states_diff):
        self.states_diff = jnp.array(states_diff)

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
    def backwardOutput(self, inputs, params):
        '''
        Backward denoise process at step t: including PQC and random measurement
        Args:
        inputs: the input data set at step t
        params: the parameters for the PQC
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params)

        # perform measurement
        output = self.randomMeasure(output_full)
        return output
    

    def backDataGeneration(self, inputs_T, params, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = [inputs_T]
        input_tplus1 = jnp.concatenate([inputs_T, jnp.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                           dtype=jnp.complex64)], axis=1)
        
        output = self.backwardOutput(input_tplus1, params)
        states.append(output)
        states = jnp.stack(states)[::-1]

        return states
