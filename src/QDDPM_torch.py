import numpy as np
import ot
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group
import torch
import torch.nn as nn
from torch.linalg import matrix_power
from opt_einsum import contract
from functools import partial
from itertools import combinations

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

class DiffusionModel(nn.Module):
    def __init__(self, n, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
    
    def HaarSampleGeneration(self, Ndata, seed):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2**self.n, size=Ndata)[:,:,0]
        return torch.from_numpy(states_T).cfloat()
    
    def scrambleCircuit_t(self, t, input, phis, gs=None):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        c = tc.Circuit(self.n, inputs=input)
        for tt in range(t):
            # single qubit rotations
            for i in range(self.n):
                c.rz(i, theta=phis[3*self.n*tt+i])
                c.ry(i, theta=phis[3*self.n*tt+self.n+i])
                c.rz(i, theta=phis[3*self.n*tt+2*self.n+i])
            # homogenous RZZ on every pair of qubits (n>=2)
            if self.n >= 2:
                for i, j in combinations(range(self.n), 2):
                    c.rzz(i, j, theta=gs[tt]/(2*np.sqrt(self.n)))
        return c.state()
    
    def set_diffusionData_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        # set single-qubit rotation angles
        np.random.seed(seed)
        phis = torch.rand(self.Ndata, 3*self.n*t)*np.pi/4. - np.pi/8.
        phis = phis*(diff_hs.repeat(3*self.n))
        if self.n > 1:
            # set homogenous RZZ gate angles
            gs = torch.rand(self.Ndata, t)*0.2 + 0.4
            gs *= diff_hs
        states = torch.zeros((self.Ndata, 2**self.n)).cfloat()
        for i in range(self.Ndata):
            if self.n > 1:
                states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i], gs[i])
            else:
                states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i])
        return states


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


class QDDPM(nn.Module):
    def __init__(self, n, na, T, L):
        '''
        the QDDPM model: backward process only work on cpu
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
        self.backCircuit_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0)

    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).cfloat()

    def randomMeasure(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        m_probs = (torch.abs(inputs.reshape(inputs.shape[0], 2**self.na, 2**self.n))**2).sum(dim=2)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze() # measurment results
        indices = 2**self.n * m_res.view(-1, 1) + torch.arange(2**self.n)
        post_state = torch.gather(inputs, 1, indices)
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        return 1./norms * post_state

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
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot)).cfloat()
        self.input_tplus1[:,:2**self.n] = inputs_T
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt])
        return self.input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = torch.zeros((self.T+1, Ndata, 2**self.n_tot)).cfloat()
        states[-1, :, :2**self.n] = inputs_T
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, -1, -1):
                states[tt, :, :2**self.n] = self.backwardOutput_t(states[tt+1], params_tot[tt])
        return states


def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set1))**2)
    r22 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set2.conj(), Set2))**2)
    r12 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set2))**2)
    return 2*r12 - r11 - r22


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - torch.abs(Set1.conj() @ Set2.T)**2.
    emt = torch.empty(0)
    Wass_dis = ot.emd2(emt, emt, M=D)
    return Wass_dis


def sinkhornDistance(Set1, Set2, reg=0.005, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - torch.abs(Set1.conj() @  Set2.T)**2.
    emt = torch.empty(0)
    if log == True:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, method='sinkhorn_log')
    else:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg)
    return sh_dis
