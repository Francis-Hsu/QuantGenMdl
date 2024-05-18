import datetime

import jax
from jax import numpy as jnp
from jax import config

import tensorcircuit as tc

from functools import partial
from itertools import combinations

config.update("jax_enable_x64", True)

K = tc.set_backend("jax")
tc.set_dtype('complex64')


def scrambleCircuitOneQubit(input, phis):
    '''
    Obtain the state through diffusion step t
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

def scrambleCircuitMultiQubit(input, phis, gs, n):
    '''
    obtain the state through diffusion step t
    Args:
    t: diffusion step
    input: the input quantum state
    phis: the single-qubit rotation angles in diffusion circuit
    gs: the angle of RZZ gates in diffusion circuit when n>=2
    '''
    t = phis.shape[0] // (3*n)
    c = tc.Circuit(n, inputs=input)

    for s in range(t):
        for i in range(n):
            c.rz(i, theta=phis[3*n*s + i])
            c.ry(i, theta=phis[3*n*s + n + i])
            c.rz(i, theta=phis[3*n*s + 2*n + i])
        
        for i, j in combinations(range(n), 2):
            c.rzz(i, j, theta=gs[s] / (2 * n ** 0.5))
    
    return c.state()


def setDiffusionDataOneQubit(inputs, diff_hs):
    '''
    Obtain the quantum data set for 1 qubit through diffusion step t
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
    phis = jax.random.uniform(key, shape=(
        Ndata, 3 * t), minval=-jnp.pi / 8., maxval=jnp.pi / 8.)
    phis *= diff_hs

    states = K.vmap(scrambleCircuitOneQubit,
                    vectorized_argnums=(0, 1))(inputs, phis)

    return states

def setDiffusionDataMultiQubit(inputs, diff_hs, n):
    '''
    Obtain the quantum data set for multi qubit through diffusion step t
    Args:
    t: diffusion step
    inputs: the input quantum data set
    diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
    n: number of qubits
    '''
    t = diff_hs.shape[0]
    Ndata = inputs.shape[0]
    
    key = jax.random.PRNGKey(t)
    phis = jax.random.uniform(key, shape=(Ndata, 3 * n * t), minval=-jnp.pi/8., maxval=jnp.pi/8.)
    phis *= jnp.repeat(diff_hs, 3*n)

    gs = jax.random.uniform(key, shape=(Ndata, t), minval=0.4, maxval=0.6)
    gs *= diff_hs 

    states = K.vmap(partial(scrambleCircuitMultiQubit, n=n), vectorized_argnums=(0, 1, 2))(inputs, phis, gs)

    return states

def unitary(key, n, shape=()):
    '''
    Sample uniformly from the unitary group
    key:  a PRNG key used as the random key.
    n: an integer indicating the resulting dimension.
    shape: optional, the batch dimensions of the result.
    '''
    a, b = jax.random.normal(key, (2, *shape, n, n))
    z = a + b * 1j
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, 0, -2, -1)

    return jax.lax.mul(q, jax.lax.expand_dims(jax.lax.div(d, abs(d).astype(d.dtype)), [-2]))


def HaarSampleGeneration(Ndata, dim, seed):
    '''
    Generate random Haar states,
    used as inputs in the t=T step for backward denoise
    Args:
    Ndata: number of samples in dataset
    '''
    key = jax.random.PRNGKey(seed)
    states_T = unitary(key, dim, (Ndata, ))[:, :, 0]

    return states_T


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

        self.backCircuit_vmap = K.jit(
            K.vmap(partial(self.backCircuit), vectorized_argnums=0))

    def backCircuit(self, input, params):
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
        L = self.L
        n_tot = self.n_tot
        c = tc.Circuit(n_tot, inputs=input)

        for l in range(L):
            for i in range(n_tot):
                c.rx(i, theta=params[2 * n_tot * l + i])
                c.ry(i, theta=params[2 * n_tot * l + n_tot + i])

            for i in range(n_tot // 2):
                c.cz(2 * i, 2 * i + 1)

            for i in range((n_tot-1) // 2):
                c.cz(2 * i + 1, 2 * i + 2)

        return c.state()

    def set_diffusionSet(self, states_diff):
        self.states_diff = states_diff

    @partial(jax.jit, static_argnums=(0, ))
    def randomMeasure(self, inputs, key):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        key: key for JAX's pseudo-random number generator
        '''
        n_batch = inputs.shape[0]
        m_probs = jnp.abs(jnp.reshape(
            inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
        m_probs = jnp.log(jnp.sum(m_probs, axis=2))
        m_res = jax.random.categorical(key, m_probs)
        indices = 2 ** self.n * \
            jnp.reshape(m_res, [-1, 1]) + jnp.arange(2 ** self.n)
        post_state = jnp.take_along_axis(inputs, indices, axis=1)
        post_state /= jnp.linalg.norm(post_state, axis=1)[:, jnp.newaxis]

        return post_state

    @partial(jax.jit, static_argnums=(0, ))
    def backwardOutput_t(self, inputs, params, key):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        key: key for JAX's pseudo-random number generator
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params)

        # perform measurement
        output_t = self.randomMeasure(output_full, key)

        return output_t

    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        '''
        Prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tot: all circuit parameters till step t+1
        '''
        # create a key for PRNG
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        zero_shape = 2 ** self.n_tot - 2 ** self.n
        zero_tensor = jnp.zeros(shape=(Ndata, zero_shape), dtype=jnp.complex64)
        input_t_plus_1 = jnp.concatenate([inputs_T, zero_tensor], axis=1)
        for tt in range(self.T - 1, t, -1):
            key, subkey = jax.random.split(key)
            output = self.backwardOutput_t(input_t_plus_1, params_tot[tt], subkey)
            input_t_plus_1 = jnp.concatenate([output, zero_tensor], axis=1)

        return input_t_plus_1

    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        # create a key for PRNG
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        states = [inputs_T]
        zero_shape = 2 ** self.n_tot - 2 ** self.n
        zero_tensor = jnp.zeros(shape=(Ndata, zero_shape), dtype=jnp.complex64)
        input_t_plus_1 = jnp.concatenate([inputs_T, zero_tensor], axis=1)
        for tt in range(self.T-1, -1, -1):
            key, subkey = jax.random.split(key)
            output = self.backwardOutput_t(
                input_t_plus_1, params_tot[tt], subkey)
            input_t_plus_1 = jnp.concatenate([output, zero_tensor], axis=1)
            states.append(output)
        states = jnp.stack(states[::-1])

        return states
