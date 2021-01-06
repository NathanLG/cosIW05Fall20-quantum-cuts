from qiskit import Aer, IBMQ, execute, QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

import copy
import numpy as np

from utils.cutter import cut_circuit
from utils.helper_fun import *

import difflib

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

def time_wise_naked_subcircs(circ, offset_number, cut_points, first_half_qb, second_half_qb):
    dag = circuit_to_dag(circ)

    offset = 0
    cut_idx = 0
    cut_qi, cut_index = cut_points[0]
    first_last_half = list(dag.nodes_on_wire(wire=dag.wires[cut_qi], only_ops=True))[cut_index]

    first_last_halves = [list(dag.nodes_on_wire(wire=dag.wires[qi], only_ops=True))[idx] for qi,idx in cut_points]
    
    first = True
    first_qasm = "OPENQASM 2.0;\n" + "include \"qelib1.inc\";\nqreg q[" + str(first_half_qb) + "];\n"
    second_qasm = "OPENQASM 2.0;\n" + "include \"qelib1.inc\";\nqreg q[" + str(second_half_qb) + "];\n"
    
    for n in dag.topological_op_nodes():
        if (n in first_last_halves):
            first = False
            offset = offset_number
        
        line = n.op.qasm()
        line += " "
        for qarg in n.qargs:
            line += "q[" + str(qarg.index - offset) + "],"
        line = line[:-1]
        line += ";"
        
        if first:
            first_qasm += line + "\n"
        else:
            second_qasm += line + "\n"

    first_qasm = first_qasm[:-1]
    second_qasm = second_qasm[:-1]

    return [QuantumCircuit.from_qasm_str(first_qasm), QuantumCircuit.from_qasm_str(second_qasm)]

def mixed_naked_subcircs(circ, offset_number, cut_points, first_half_qb, second_half_qb):
    dag = circuit_to_dag(circ)

    num_space = 0
    for _,_,t in cut_points:
        if t == "space":
            num_space += 1

    offset = 0
    cut_idx = 0

    first_last_halves = {list(dag.nodes_on_wire(wire=dag.wires[qi], only_ops=True))[idx]:t for qi,idx,t in cut_points}

    space_cut_indeces = {}
    idx = 0
    for n in first_last_halves:
        if first_last_halves[n] == "space":
            space_cut_indeces[n] = idx
            idx += 1
    
    first = True
    first_tmp = False
    first_qasm = "OPENQASM 2.0;\n" + "include \"qelib1.inc\";\nqreg q[" + str(first_half_qb) + "];\n"
    
    second_qasms = ["OPENQASM 2.0;\n" + "include \"qelib1.inc\";\nqreg q[" + str(second_half_qb) + "];\n" for i in range(2**num_space)]
    
    for n in dag.topological_op_nodes():
        if (n in first_last_halves):
            first = False
            offset = offset_number
            if first_last_halves[n] == "space":
                # Enumerate possibilities for second half of space-wise cut
                space_idx = space_cut_indeces[n]
                for i in range(2**num_space):
                    if (int(i/(2**space_idx)) % 2 == 1):
                        idx = n.qargs[0].index
                        for qarg in n.qargs:
                            if qarg.index > idx:
                                idx = qarg.index
                        second_qasms[i] += n.op.qasm()[1:] + " " + "q[" + str(idx - offset) + "];" + "\n"
                continue
        
            
        line = n.op.qasm()
        line += " "
        for qarg in n.qargs:
            if qarg.index - offset < 0 or qarg.index - offset >= second_half_qb:
                offset = 0
                first_tmp = True
        for qarg in n.qargs:
            line += "q[" + str(qarg.index - offset) + "],"

        line = line[:-1]
        line += ";"
        
        if first_tmp:
            first_tmp = False
            offset = offset_number
            first_qasm += line + "\n"
            continue 
        if first:
            first_qasm += line + "\n"
        else:
            for i in range(2**num_space):
                second_qasms[i] += line + "\n"


    first_qasm = first_qasm[:-1]
    for second_qasm in second_qasms:
        second_qasm = second_qasm[:-1]

    return [QuantumCircuit.from_qasm_str(first_qasm), [QuantumCircuit.from_qasm_str(second_qasm) for second_qasm in second_qasms]]


def time_wise_cut(circ, size, offset_number, cut_points, first_half_qb, second_half_qb):
    subcirc1, subcirc2 = time_wise_naked_subcircs(circ, offset_number, cut_points, first_half_qb, second_half_qb)
    
    sc1_sv = evaluate_circ2(subcirc1)
    sc1_svs = []

    non_cut_size = first_half_qb - len(cut_points)
    sc1_size = 2**non_cut_size

    for i in range(2**len(cut_points)):
        sc1_svs.append(sc1_sv[i * sc1_size : (i + 1) * sc1_size])

    sc2_svs = []
    for i in range(2**len(cut_points)):
        num = i
        sc2_pre = QuantumCircuit(second_half_qb)
        for p in range(len(cut_points)):
            if (num % 2 == 1):
                sc2_pre.x(p)
            num = int(num/2)
        sc2_svs.append(evaluate_circ2(sc2_pre.compose(subcirc2)))
        
    reconstructed = np.zeros(2**size, dtype=np.cdouble)
    for i in range(2**len(cut_points)):
        reconstructed += np.kron(sc2_svs[i], sc1_svs[i])
    
    return reconstructed

def mixed_cut(circ, size, offset_number, cut_points, first_half_qb, second_half_qb):
    subcirc1, subcirc2s = mixed_naked_subcircs(circ, offset_number, cut_points, first_half_qb, second_half_qb)
    
    num_space = 0
    num_time = 0
    space_pos = {}
    time_pos = {}

    for q,_,t in cut_points:
        if t == "space":
            space_pos[num_space] = q
            num_space += 1
        elif t == "time":
            time_pos[num_time] = q - offset_number
            num_time += 1
        else:
            assert(False)

    if num_space + num_time != len(cut_points):
        assert(False)

    order = [t for _,_,t in cut_points]
    
    sc1_sv = evaluate_circ2(subcirc1)

    sc1_sv_s = []

    for idx in range(2**num_space):
        mask = [1 for i in range(len(sc1_sv))]
        for p in range(num_space):
            mask = [mask[i] * ((int(i/2**(space_pos[p])) + ((idx + 1) % 2)) % 2) for i in range(len(sc1_sv))]
            #sc1_sv_s.append([sc1_sv[i] * ((int(i/2**(space_pos[p])) + ((idx + 1) % 2)) % 2) for i in range(len(sc1_sv))])
            idx = int(idx/2)
        sc1_sv_s.append([sc1_sv[i] * mask[i] for i in range(len(sc1_sv))])

    sc1_svs = []
    

    non_cut_size = first_half_qb - len(cut_points) + num_space
    sc1_size = 2**non_cut_size
    
    for sp_idx in range(2**num_space):
        for i in range(2**(num_time)):#len(cut_points) - num_space)):
            sc1_svs.append(sc1_sv_s[sp_idx][sc1_size * i:sc1_size * (i + 1)])

    sc2_svs = []
    for i in range(2**len(cut_points)):
        t_num = i
        s_num = int(i/2**num_time)#(len(cut_points) - num_space))
        sc2_pre = QuantumCircuit(second_half_qb)
        for p in range(num_time):#len(cut_points) - num_space):
            if t_num % 2 == 1:
                sc2_pre.x(time_pos[p])
            t_num = int(t_num/2)
        # print(sc2_pre.compose(subcirc2s[s_num]))
        sc2_svs.append(evaluate_circ2(sc2_pre.compose(subcirc2s[s_num])))
    
    reconstructed = np.zeros(2**size, dtype=np.cdouble)

    for i in range(2**len(cut_points)):
        reconstructed += np.kron(sc2_svs[i], sc1_svs[i])
  
    return reconstructed
