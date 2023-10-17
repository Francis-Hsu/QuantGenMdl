from functools import partial

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

import jax
from jax import numpy as jnp

import tensorcircuit as tc

np_config.enable_numpy_behavior()

K = tc.set_backend("jax")
tc.set_dtype('complex64')

# jit or not?
def scrambleCircuitOneQubit(input, phis):
    '''
    obtain the state through diffusion step t
    Args:
    t: diffusion step
    input: the input quantum state
    phis: the single-qubit rotation angles in diffusion circuit
    gs: the angle of RZZ gates in diffusion circuit when n>=2
    '''
    # input, phis = params
    t = phis.shape[0] // 3
    c = tc.Circuit(1, inputs=input)

    for s in range(t):
        # single qubit rotations
        c.rz(0, theta=phis[3 * s])
        c.ry(0, theta=phis[3 * s + 1])
        c.rz(0, theta=phis[3 * s + 2])

    return c.state()

def setDiffusionDataOneQubit(inputs, diff_hs):
    '''
    obtain the quantum data set for 1 qubit through diffusion step t
    Args:
    t: diffusion step
    inputs: the input quantum data set
    diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
    '''
    t = diff_hs.shape[0]
    Ndata = inputs.shape[0]
    diff_hs = jnp.repeat(diff_hs, 3)

    # set single-qubit rotation angles
    key = jax.random.PRNGKey(t)
    phis = jax.random.uniform(key, shape=(Ndata, 3 * t), minval=-jnp.pi / 8., maxval=jnp.pi / 8.)
    phis *= diff_hs

    states = K.vmap(scrambleCircuitOneQubit, vectorized_argnums=(0, 1))(inputs, phis)

    return states

def HaarSampleGeneration(Ndata, seed):
    '''
    generate random haar states,
    used as inputs in the t=T step for backward denoise
    Args:
    Ndata: number of samples in dataset
    '''
    np.random.seed(seed)
    states_T = unitary_group.rvs(dim=2, size=Ndata)[:,:,0]

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

class QDDPM():
    def __init__(self, n, na, T, L):
        '''
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=self.L), vectorized_argnums=0))

    def set_diffusionSet(self, states_diff):
        self.states_diff = tf.convert_to_tensor(states_diff)

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
    def backwardOutput_t(self, inputs, params):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params)

        # perform measurement
        output_t = self.randomMeasure(output_full)

        return output_t
    
    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tot: all circuit parameters till step t+1
        '''
        zero_tensor = jnp.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), dtype=jnp.complex64)
        input_tplus1 = jnp.concatenate([inputs_T, zero_tensor], axis=1)
        params_tot = tf.constant(params_tot, dtype=tf.float32)
        for tt in range(self.T-1, t, -1):
            output = self.backwardOutput_t(input_tplus1, params_tot[tt])
            input_tplus1 = jnp.concatenate([output, zero_tensor], axis=1)

        return input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = [inputs_T]
        input_tplus1 = jnp.concatenate([inputs_T, tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                           dtype=jnp.complex64)], axis=1)
        params_tot = tf.cast(tf.convert_to_tensor(params_tot), dtype=tf.float32)
        for tt in range(self.T-1, -1, -1):
            output = self.backwardOutput_t(input_tplus1, params_tot[tt])
            input_tplus1 = jnp.concatenate([output, jnp.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                              dtype=jnp.complex64)], axis=1)
            states.append(output)
        states = jnp.stack(states)[::-1]

        return states
