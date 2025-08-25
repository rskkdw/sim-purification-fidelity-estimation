# %%
import multiprocessing as mp
import stim
import os, sys, argparse, json
from typing import Callable, List, Dict, Any, Set, FrozenSet, Iterable, Tuple
import numpy as np
import sinter
#mp.set_start_method('fork', force=True)


def generate_bell_pair(p:float=0.1, qubit_index_start:int=0):
    a, b = 0,1
    qubits = [a,b]
    q2i: Dict[complex, int] = {q: i for i, q in enumerate(qubits, start=qubit_index_start)}

    circuit = stim.Circuit()
    circuit.append_operation("R", [q2i[a], q2i[b]])
    circuit.append_operation("DEPOLARIZE1", [q2i[a], q2i[b]], p)
    circuit.append_operation("H", [q2i[a]])
    circuit.append_operation("DEPOLARIZE1", [q2i[a]], p)
    circuit.append_operation("CNOT", [q2i[a], q2i[b]])
    circuit.append_operation("DEPOLARIZE2", [q2i[a], q2i[b]], p)
    return circuit


def rec_distillation(circuit, qubits, p):
    q2i: Dict[complex, int] = {q: i for q, i in enumerate(qubits)}

    circuit.append_operation("CNOT", [q2i[0], q2i[2]])
    circuit.append_operation("DEPOLARIZE2", [q2i[0], q2i[2]], p)
    circuit.append_operation("CNOT", [q2i[1], q2i[3]])
    circuit.append_operation("DEPOLARIZE2", [q2i[1], q2i[3]], p)
    circuit.append_operation("X_ERROR", [q2i[2], q2i[3]], p)
    circuit.append_operation("M", [q2i[2], q2i[3]])
    circuit.append_operation("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)], [0, 0, 0, 1])

    circuit.append_operation("CNOT", [q2i[6], q2i[4]])
    circuit.append_operation("DEPOLARIZE2", [q2i[6], q2i[4]], p)
    circuit.append_operation("CNOT", [q2i[7], q2i[5]])
    circuit.append_operation("DEPOLARIZE2", [q2i[7], q2i[5]], p)
    circuit.append_operation("Z_ERROR", [q2i[6], q2i[7]], p)
    circuit.append_operation("MX", [q2i[6], q2i[7]])
    circuit.append_operation("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)], [0, 0, 0, 2])


    circuit.append_operation("CNOT", [q2i[4], q2i[0]])
    circuit.append_operation("DEPOLARIZE2", [q2i[4], q2i[0]], p)
    circuit.append_operation("CNOT", [q2i[5], q2i[1]])
    circuit.append_operation("DEPOLARIZE2", [q2i[5], q2i[1]], p)
    circuit.append_operation("H", [q2i[4], q2i[5]])
    circuit.append_operation("DEPOLARIZE1", [q2i[4], q2i[5]], p)
    circuit.append_operation("Z_ERROR", [q2i[4], q2i[5]], p)
    circuit.append_operation("S_DAG", [q2i[4], q2i[5]])
    circuit.append_operation("H", [q2i[4], q2i[5]])
    circuit.append_operation("X", [q2i[4]])
    circuit.append_operation("MZ", [q2i[4], q2i[5]])
    circuit.append_operation("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)], [0, 0, 0, 3])

    return circuit

def generate_circuit(code_efficiency: int, distillation_efficiency: int, 
                     code0: str, code1: str, direc: str, bell_pair_infidelity: float, error_type: str,
                     depth:int, p:float=0.1, qubit_index_start:int=0):
    num_qubit = 2 * code_efficiency * distillation_efficiency**(depth)
    circuit = stim.Circuit()
    circuit.append_operation("R", list(range(num_qubit)))
    #circuit.append_operation("DEPOLARIZE1", list(range(num_qubit)), p)
    circuit.append_operation("H", [i for i in range(num_qubit) if i%2==0])
    #circuit.append_operation("DEPOLARIZE1", list(range(num_qubit)), p)
    circuit.append_operation("CNOT", list(range(num_qubit)))
    circuit.append_operation("DEPOLARIZE2", list(range(num_qubit)), bell_pair_infidelity)

    for i in range(depth):
        jump = 2 * code_efficiency * distillation_efficiency**(i)
        current = 0
        while current < num_qubit:
            rec_distillation(circuit, [current, current+1, current+jump, current+jump+1, 
            current+jump*2, current+jump*2+1, current+jump*3, current+jump*3+1], p)
            #if i < depth:
            #    circuit.append_operation("H", [current, current+1])
            #    circuit.append_operation("DEPOLARIZE1", [current, current+1], p)
            current += jump*4
    
    if error_type == "Z":
        circuit.append_operation("H", [0,1])
    circuit.append_operation("M", [0,1])
    circuit.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2)])

    return circuit

def generate_circuits(depths: List, ps: List, code_efficiency: int, distillation_efficiency: int, 
                     code0: str, code1: str, direc: str, bell_pair_infidelity: float, error_type: str):
    circuits = []
    for depth in depths:
        for p in ps:
            circuit = generate_circuit(code_efficiency, distillation_efficiency, code0, code1, direc, 
                                        bell_pair_infidelity, error_type, depth, p)
            filename_stim = direc+f"f={bell_pair_infidelity},p={p},r={depth},code={code0}-{code1},errortype={error_type}.stim"
            with open(filename_stim, 'w') as fd:
                circuit.to_file(fd)
            filename_dem = direc+f"f={bell_pair_infidelity},p={p},r={depth},code={code0}-{code1},errortype={error_type}.dem"
            dem = circuit.detector_error_model(allow_gauge_detectors=True)
            print(filename_dem)
            with open(filename_dem, 'w') as fd:
                detector_error_model.to_file(fd)
            circuits.append(circuit)
    return circuits

def main():
    # パーサーの作成
    parser = argparse.ArgumentParser(description='Generate circuits for error correction.')

    # 引数の追加
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')

    # 引数の解析
    args = parser.parse_args()

    # 引数の取得
    config_path = args.config

    # JSONファイルから設定を読み込む
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 設定の取得
    depths = config.get('depths', [0, 1, 2, 3])#, 4, 5, 6])
    ps = config.get('ps', [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)])
    code_efficiency = config.get('code_efficiency', 1)
    distillation_efficiency = config.get('distillation_efficiency', 2)
    code0 = config.get('code0', 'phys')
    code1 = config.get('code1', 'phys')
    direc = config.get('direc', 'out')
    bell_pair_infidelity = config.get('bell_pair_infidelity', 0.16)
    error_type = config.get('error_type', 'X')

    def bitpack(bits):
        result = 0
        for bit in reversed(bits):
            result = (result << 1) | (bit & 1)
        return result
    def make_mask(bits):
        results = []
        while bits:
            bits_ = bits[:8]
            bits = bits[8:]
            result = bitpack(bits_)
            results.append(result)
        return np.array(results, dtype=np.uint8)

    tasks = []
    for basis, coords_num, mask_base in [('X',2, [0,1,0]), ('Y',3,[0,0,1]), ('Z',1, [1,0,0]), ('all', None,[1,1,1])]:
        print("Generating sinter tasks for", basis, "...")
        tasks_ = [
            sinter.Task(
                circuit=circ,
                postselection_mask=make_mask(mask_base * (circ.detector_error_model().num_detectors // 3)),
                json_metadata={'p': p, 'depth': depth, 'post_selection_basis': basis, 'errortype': error_type, 'bell_pair_infidelity': bell_pair_infidelity},
            )
            for depth in [1, 2, 3, 4, 5]
            for p in [10**(-1), 10**(-2), 10**(-3), 10**(-4)]
            for bell_pair_infidelity in [0.16, 0.3]
            if (circ :=generate_circuit(
                    code_efficiency=1,
                    distillation_efficiency=4,
                    code0="phys",
                    code1="phys",
                    direc="out/",
                    bell_pair_infidelity=bell_pair_infidelity,
                    error_type="X",
                    depth=depth,
                    p=p,
                ) ) 
        ]
        tasks.extend(tasks_)

    print("collecting sinter results...")
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=12,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=100_000_000,
        max_errors=10000,
        print_progress=True,
        save_resume_filepath="out/sinter_resume.json",
        count_detection_events=True,
    )



if __name__ == '__main__':
    main()




# %%
