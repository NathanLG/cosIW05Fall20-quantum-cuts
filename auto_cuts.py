from qiskit import Aer, IBMQ, execute, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
import numpy as np

from nathan_util import *
from utils.helper_fun import *

import time
#print millis

def threshold(x):
    if np.absolute(x) < 10**(-15):
        return 0. + 0.j
    return x

def evaluate_circ2(circ):
    # print('using statevector simulator')
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend=backend, optimization_level=0)
    result = job.result()
    outputstate = np.array(result.get_statevector(circ))
    outputstate = np.vectorize(threshold)(outputstate)

    return outputstate


def bv_time_subcircuit_1(size):
    subcirc = QuantumCircuit(size)
    subcirc.x(size - 1)
    subcirc.h(size - 1)
    for i in range(size - 1):
        subcirc.h(i)
        subcirc.cx(i, size - 1)
        subcirc.h(i)

    print(subcirc)
    return subcirc

def bv_time_subcircuit_2(size, one_init):
    subcirc = QuantumCircuit(size)
    if one_init:
        subcirc.x(0)
    for i in range(1,size):
        subcirc.h(i)
        subcirc.cx(i,0)
        subcirc.h(i)
    subcirc.h(0)

    print(subcirc)
    return subcirc

def cut_bv(size):
    if size % 2 == 0:
        answer_qbit = int(size/2) - 1
        sc_1 = bv_time_subcircuit_1(int(size / 2))
        sc_2_0 = bv_time_subcircuit_2(int(size/2 + 1), False)
        sc_2_1 = bv_time_subcircuit_2(int(size/2 + 1), True)
    else:
        answer_qbit = int(size/2)
        sc_1 = bv_time_subcircuit_1(int(size / 2 + 1))
        sc_2_0 = bv_time_subcircuit_2(int(size/2 + 1), False)
        sc_2_1 = bv_time_subcircuit_2(int(size/2 + 1), True)

    sv_1 = evaluate_circ2(sc_1)
    sv_2_0 = evaluate_circ2(sc_2_0)
    sv_2_1 = evaluate_circ2(sc_2_1)
    
    reconstructed = np.kron(sv_2_0, sv_1[:int(len(sv_1)/2)]) + np.kron(sv_2_1, sv_1[int(len(sv_1)/2):])

    reconstructed = bit_permute(reconstructed, [x for x in range(answer_qbit)] + [x + 1 for x in range(answer_qbit, size - 1)] + [answer_qbit])

    return reconstructed

def tw_sl_4():
    sc1 = QuantumCircuit(2)
    
    sc1.h(0)
    sc1.h(1)
    sc1.cz(0,1)
    sc1.rx(np.pi/2, 0)
    sc1.rx(np.pi/2, 1)
    sc1.t(0)
    sc1.h(0)

    sc1_sv = evaluate_circ2(sc1)
    
    sc2_0 = QuantumCircuit(3)
    sc2_1 = QuantumCircuit(3)
    sc2_1.x(0)
    
    for sc in [sc2_0, sc2_1]:
        sc.h(1)
        sc.t(1)
        sc.h(2)
        sc.t(2)
        sc.cz(0,1)
        sc.rx(np.pi/2, 0)
        sc.ry(np.pi/2, 1)
        sc.t(0)
        sc.h(0)
        sc.cz(1,2)
        sc.rx(np.pi/2, 1)
        sc.rx(np.pi/2, 2)
        sc.t(1)
        sc.h(1)
        sc.t(2)
        sc.h(2)

    sc2_0_sv = evaluate_circ2(sc2_0)
    sc2_1_sv = evaluate_circ2(sc2_1)

    print(sc1)
    print(sc2_0)
    print(sc2_1)

    reconstructed = np.kron(sc2_0_sv, sc1_sv[:2]) + np.kron(sc2_1_sv, sc1_sv[2:])
    return reconstructed

def sw_sl_4():
    P0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
    P1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])

    sc1 = QuantumCircuit(2)
    sc1.h(0)
    sc1.h(1)
    sc1.cz(0,1)
    sc1.ry(np.pi/2, 0)
    sc1.ry(np.pi/2, 1)
    sc1.t(0)
    sc1.h(0)
    sc1.ry(np.pi/2, 1)
    sc1.t(1)
    sc1.h(1)
    print(sc1)
    sc1_sv = evaluate_circ2(sc1)
    sc1_sv0 = np.matmul(sc1_sv, P0)
    sc1_sv1 = np.matmul(sc1_sv, P1)

    sc2_i = QuantumCircuit(2)
    sc2_z = QuantumCircuit(2)
    for sc in [sc2_i, sc2_z]:
        sc.h(0)
        sc.h(1)
        sc.t(0)
        sc.t(1)
        if sc == sc2_z:
            sc.z(0)
        sc.rx(np.pi/2, 0)
        sc.cz(0,1)
        sc.rx(np.pi/2, 0)
        sc.rx(np.pi/2, 1)
        sc.t(0)
        sc.t(1)
        sc.h(0)
        sc.h(1)

    print(sc2_i)
    print(sc2_z)
    sc2_i_sv = evaluate_circ2(sc2_i)
    sc2_z_sv = evaluate_circ2(sc2_z)

    reconstructed = np.kron(sc2_i_sv, sc1_sv0) + np.kron(sc2_z_sv, sc1_sv1)
    return reconstructed


def sl():
    c = QuantumCircuit(4)
    c.h(0)
    c.h(1)
    c.h(2)
    c.h(3)
    c.cz(0,1)
    c.t(2)
    c.t(3)
    c.rx(np.pi/2, 0)
    c.rx(np.pi/2, 1)
    c.t(0)
    c.h(0)
    c.cz(1,2)
    c.rx(np.pi/2, 1)
    c.ry(np.pi/2, 2)
    c.t(1)
    c.h(1)
    c.cz(2,3)
    c.rx(np.pi/2, 2)
    c.rx(np.pi/2, 3)
    c.t(2)
    c.h(2)
    c.t(3)
    c.h(3)
    return c


def aqft_auto_timecut_a(size):
    if size < 6:
        return

    each_side = int((size - 4)/2)

    #print("Subcircuit 1")
    sc1_size = int(each_side + 3)
    sc1 = QuantumCircuit(each_side + 3)

    sc1.h(0)
    for i in range(each_side):
        sc1.cu1(np.pi/2, i, i + 1)
        sc1.cu1(np.pi/4, i, i + 2)
        sc1.h(i + 1)
        sc1.cu1(np.pi/8, i, i + 3)
    
    sc1.cu1(np.pi/2, sc1_size - 3, sc1_size - 2)
    sc1.cu1(np.pi/4, sc1_size - 3, sc1_size - 1)
    sc1.h(sc1_size - 2)

    sc1_sv = evaluate_circ2(sc1)
    my_size = 2**each_side
    sc1_000_sv = sc1_sv[:my_size]
    sc1_001_sv = sc1_sv[my_size: my_size * 2]
    sc1_010_sv = sc1_sv[my_size * 2: my_size * 3]
    sc1_011_sv = sc1_sv[my_size * 3: my_size * 4]
    sc1_100_sv = sc1_sv[my_size * 4: my_size * 5]
    sc1_101_sv = sc1_sv[my_size * 5: my_size * 6]
    sc1_110_sv = sc1_sv[my_size * 6: my_size * 7]
    sc1_111_sv = sc1_sv[my_size * 7:]
    sc1_svs = [sc1_000_sv, sc1_001_sv, sc1_010_sv, sc1_011_sv,
               sc1_100_sv, sc1_101_sv, sc1_110_sv, sc1_111_sv]

    #print(sc1)

    #print("Subcircuit 2")
    sc2_size = int(each_side + 4)

    sc2_000 = QuantumCircuit(sc2_size)
    sc2_001 = QuantumCircuit(sc2_size)
    sc2_010 = QuantumCircuit(sc2_size)
    sc2_011 = QuantumCircuit(sc2_size)
    sc2_100 = QuantumCircuit(sc2_size)
    sc2_101 = QuantumCircuit(sc2_size)
    sc2_110 = QuantumCircuit(sc2_size)
    sc2_111 = QuantumCircuit(sc2_size)
    
    for sc2 in [sc2_001, sc2_011, sc2_101, sc2_111]:
        sc2.x(0)
        
    for sc2 in [sc2_010, sc2_011, sc2_110, sc2_111]:
        sc2.x(1)
        
    for sc2 in [sc2_100, sc2_101, sc2_110, sc2_111]:
        sc2.x(2)
    
    for sc2 in [sc2_000, sc2_001, sc2_010, sc2_011, sc2_100, sc2_101, sc2_110, sc2_111]:
        sc2.cu1(np.pi/8, 0, 3)
        for i in range(1,each_side + 1):
            sc2.cu1(np.pi/2, i, i + 1)
            sc2.cu1(np.pi/4, i, i + 2)
            sc2.h(i + 1)
            sc2.cu1(np.pi/8, i, i + 3)
        
        sc2.cu1(np.pi/2, sc2_size - 3, sc2_size - 2)
        sc2.cu1(np.pi/4, sc2_size - 3, sc2_size - 1)
        sc2.h(sc2_size - 2)
        sc2.cu1(np.pi/2, sc2_size - 2, sc2_size - 1)
        sc2.h(sc2_size - 1)

    sc2_000_sv = evaluate_circ2(sc2_000)
    sc2_001_sv = evaluate_circ2(sc2_001)
    sc2_010_sv = evaluate_circ2(sc2_010)
    sc2_011_sv = evaluate_circ2(sc2_011)
    sc2_100_sv = evaluate_circ2(sc2_100)
    sc2_101_sv = evaluate_circ2(sc2_101)
    sc2_110_sv = evaluate_circ2(sc2_110)
    sc2_111_sv = evaluate_circ2(sc2_111)
    sc2_svs = [sc2_000_sv, sc2_001_sv, sc2_010_sv, sc2_011_sv,
               sc2_100_sv, sc2_101_sv, sc2_110_sv, sc2_111_sv]

    reconstructed = np.zeros(2**size, dtype=np.cdouble)
    for i in range(8):
        sc1sv = sc1_svs[i]
        sc2sv = sc2_svs[i]

        reconstructed += np.kron(sc2sv, sc1sv)

    return reconstructed
    
def aqft_auto_mixcut_a(size):
    if size < 6:
        return

    each_side = int((size - 4)/2)

    #print("Subcircuit 1")
    sc1_size = int(each_side + 3)
    sc1 = QuantumCircuit(each_side + 3)

    sc1.h(0)
    for i in range(each_side):
        sc1.cu1(np.pi/2, i, i + 1)
        sc1.cu1(np.pi/4, i, i + 2)
        sc1.h(i + 1)
        sc1.cu1(np.pi/8, i, i + 3)
    
    sc1.cu1(np.pi/2, sc1_size - 3, sc1_size - 2)
    sc1.cu1(np.pi/4, sc1_size - 3, sc1_size - 1)
    sc1.h(sc1_size - 2)

    sc1_sv = evaluate_circ2(sc1)
    
    #Proj0 = np.diag([(int(x/(2**(sc1_size - 3))) + 1) % 2 for x in range(2**sc1_size)])
    #Proj1 = np.diag([ int(x/(2**(sc1_size - 3))) % 2 for x in range(2**sc1_size)])

    #sc1_sv_0 = np.matmul(sc1_sv, Proj0)
    #sc1_sv_1 = np.matmul(sc1_sv, Proj1)
    sc1_sv_0 = [sc1_sv[i] * ((int(i/(2**(sc1_size - 3))) + 1) % 2) for i in range(len(sc1_sv))]
    sc1_sv_1 = [sc1_sv[i] * ((int(i/(2**(sc1_size - 3)))) % 2) for i in range(len(sc1_sv))]


    my_size = 2**(each_side + 1)
    sc1_sv_000 = sc1_sv_0[:my_size]                 
    sc1_sv_001 = sc1_sv_0[my_size: my_size * 2]     
    sc1_sv_010 = sc1_sv_0[my_size * 2: my_size * 3] 
    sc1_sv_011 = sc1_sv_0[my_size * 3:]             
    sc1_sv_100 = sc1_sv_1[:my_size]
    sc1_sv_101 = sc1_sv_1[my_size: my_size * 2]
    sc1_sv_110 = sc1_sv_1[my_size * 2: my_size * 3]
    sc1_sv_111 = sc1_sv_1[my_size * 3:]
    sc1_svs = [sc1_sv_000, sc1_sv_001, sc1_sv_010, sc1_sv_011,
               sc1_sv_100, sc1_sv_101, sc1_sv_110, sc1_sv_111]

#    for x in sc1_svs:
#        print(x)
    #print(sc1_sv_0, sc1_sv_1)

    #print("Subcircuit 2")
    sc2_size = int(each_side + 3)

    sc2_I00 = QuantumCircuit(sc2_size)
    sc2_I01 = QuantumCircuit(sc2_size)
    sc2_I10 = QuantumCircuit(sc2_size)
    sc2_I11 = QuantumCircuit(sc2_size)
    sc2_U00 = QuantumCircuit(sc2_size)
    sc2_U01 = QuantumCircuit(sc2_size)
    sc2_U10 = QuantumCircuit(sc2_size)
    sc2_U11 = QuantumCircuit(sc2_size)

    for sc2 in [sc2_I01, sc2_I11, sc2_U01, sc2_U11]:
        sc2.x(0)
        
    for sc2 in [sc2_I10, sc2_I11, sc2_U10, sc2_U11]:
        sc2.x(1)
        
    for sc2 in [sc2_U00, sc2_U01, sc2_U10, sc2_U11]:
        sc2.u1(np.pi/8, 2)

    for sc2 in [sc2_I00, sc2_I01, sc2_I10, sc2_I11, sc2_U00, sc2_U01, sc2_U10, sc2_U11]:
        sc2.cu1(np.pi/8, 0, 3)
        for i in range(0,each_side):
            sc2.cu1(np.pi/2, i, i + 1)
            sc2.cu1(np.pi/4, i, i + 2)
            sc2.h(i + 1)
            sc2.cu1(np.pi/8, i, i + 3)
        
        sc2.cu1(np.pi/2, sc2_size - 3, sc2_size - 2)
        sc2.cu1(np.pi/4, sc2_size - 3, sc2_size - 1)
        sc2.h(sc2_size - 2)
        sc2.cu1(np.pi/2, sc2_size - 2, sc2_size - 1)
        sc2.h(sc2_size - 1)

    sc2_I00_sv = evaluate_circ2(sc2_I00)
    sc2_I01_sv = evaluate_circ2(sc2_I01)
    sc2_I10_sv = evaluate_circ2(sc2_I10)
    sc2_I11_sv = evaluate_circ2(sc2_I11)
    sc2_U00_sv = evaluate_circ2(sc2_U00)
    sc2_U01_sv = evaluate_circ2(sc2_U01)
    sc2_U10_sv = evaluate_circ2(sc2_U10)
    sc2_U11_sv = evaluate_circ2(sc2_U11)
    sc2_svs = [sc2_I00_sv, sc2_I01_sv, sc2_I10_sv, sc2_I11_sv,
               sc2_U00_sv, sc2_U01_sv, sc2_U10_sv, sc2_U11_sv]

    reconstructed = np.zeros(2**size, dtype=np.cdouble)
    for i in range(8):
        sc1sv = sc1_svs[i]
        sc2sv = sc2_svs[i]

        reconstructed += np.kron(sc2sv, sc1sv)

    return reconstructed

def aqft6cut():
    print("Subcircuit 1")
    sc1 = QuantumCircuit(4)
    sc1.h(0)
    sc1.cu1(np.pi/2, 0, 1)
    sc1.cu1(np.pi/4, 0, 2)
    sc1.h(1)
    sc1.cu1(np.pi/8, 0, 3)
    sc1.cu1(np.pi/2, 1, 2)
    sc1.cu1(np.pi/4, 1, 3)
    sc1.h(2)

    sc1_sv = evaluate_circ2(sc1)
    sc1_000_sv = sc1_sv[:2]
    sc1_001_sv = sc1_sv[2:4]
    sc1_010_sv = sc1_sv[4:6]
    sc1_011_sv = sc1_sv[6:8]
    sc1_100_sv = sc1_sv[8:10]
    sc1_101_sv = sc1_sv[10:12]
    sc1_110_sv = sc1_sv[12:14]
    sc1_111_sv = sc1_sv[14:]
    print(sc1)
    print(sc1_sv)
    sc1_svs = [sc1_000_sv, sc1_001_sv, sc1_010_sv, sc1_011_sv,
               sc1_100_sv, sc1_101_sv, sc1_110_sv, sc1_111_sv]

    print("Subcircuit 2")
    sc2_000 = QuantumCircuit(5)
    sc2_001 = QuantumCircuit(5)
    sc2_010 = QuantumCircuit(5)
    sc2_011 = QuantumCircuit(5)
    sc2_100 = QuantumCircuit(5)
    sc2_101 = QuantumCircuit(5)
    sc2_110 = QuantumCircuit(5)
    sc2_111 = QuantumCircuit(5)

    for sc2 in [sc2_001, sc2_011, sc2_101, sc2_111]:
        sc2.x(0)

    for sc2 in [sc2_010, sc2_011, sc2_110, sc2_111]:
        sc2.x(1)

    for sc2 in [sc2_100, sc2_101, sc2_110, sc2_111]:
        sc2.x(2)

    for sc2 in [sc2_000, sc2_001, sc2_010, sc2_011, sc2_100, sc2_101, sc2_110, sc2_111]:
        sc2.cu1(np.pi/8, 0, 3)
        sc2.cu1(np.pi/2, 1, 2)
        sc2.cu1(np.pi/4, 1, 3)
        sc2.h(2)
        sc2.cu1(np.pi/8, 1, 4)
        sc2.cu1(np.pi/2, 2, 3)
        sc2.cu1(np.pi/4, 2, 4)
        sc2.h(3)
        sc2.cu1(np.pi/2, 3, 4)
        sc2.h(4)
        print(sc2)
    
    sc2_000_sv = evaluate_circ2(sc2_000)
    sc2_001_sv = evaluate_circ2(sc2_001)
    sc2_010_sv = evaluate_circ2(sc2_010)
    sc2_011_sv = evaluate_circ2(sc2_011)
    sc2_100_sv = evaluate_circ2(sc2_100)
    sc2_101_sv = evaluate_circ2(sc2_101)
    sc2_110_sv = evaluate_circ2(sc2_110)
    sc2_111_sv = evaluate_circ2(sc2_111)
    sc2_svs = [sc2_000_sv, sc2_001_sv, sc2_010_sv, sc2_011_sv,
               sc2_100_sv, sc2_101_sv, sc2_110_sv, sc2_111_sv]

    reconstructed = np.zeros(64, dtype=np.cdouble)
    for i in range(8):
        sc1sv = sc1_svs[i]
        sc2sv = sc2_svs[i]

        reconstructed += np.kron(sc2sv, sc1sv)

    return reconstructed

def aqft6cutmix():
    print("Subcircuit 1")
    sc1 = QuantumCircuit(4)
    sc1.h(0)
    sc1.cu1(np.pi/2, 0, 1)
    sc1.cu1(np.pi/4, 0, 2)
    sc1.h(1)
    sc1.cu1(np.pi/8, 0, 3)
    sc1.cu1(np.pi/2, 1, 2)
    sc1.cu1(np.pi/4, 1, 3)
    sc1.h(2)

    sc1_sv = evaluate_circ2(sc1)
    sc1_000_sv = sc1_sv[:2]
    sc1_001_sv = sc1_sv[2:4]
    sc1_010_sv = sc1_sv[4:6]
    sc1_011_sv = sc1_sv[6:8]
    sc1_100_sv = sc1_sv[8:10]
    sc1_101_sv = sc1_sv[10:12]
    sc1_110_sv = sc1_sv[12:14]
    sc1_111_sv = sc1_sv[14:]
    print(sc1)
    print(sc1_sv)
    sc1_svs = [sc1_000_sv, sc1_001_sv, sc1_010_sv, sc1_011_sv,
               sc1_100_sv, sc1_101_sv, sc1_110_sv, sc1_111_sv]


# print(cut_bv(10))
#   
# full = generate_circ(10, "bv")
# print(evaluate_circ2(full))

# full = sl()#generate_circ(4, "supremacy_linear")
# print(full)
# print(evaluate_circ2(full))
#   
# print(tw_sl_4())

#full = generate_circ(8, "aqft")
#print(full)
#dag = circuit_to_dag(full)
##dag_drawer(dag)
#  
##print(aqft6cut())
#millis = int(round(time.time() * 1000))
#for i in range(100):
#    aqft_auto_timecut_a(24)
#print(int(round(time.time() * 1000)) - millis)
##print(evaluate_circ2(full))
##  
#millis = int(round(time.time() * 1000))
#for i in range(100):
#    aqft_auto_mixcut_a(24)
#print(int(round(time.time() * 1000)) - millis)
## print(full)
