import os
import time
import numpy as np
import sympy as sp
from collections import namedtuple
from functools import reduce
from utils import *
from train_variational import get_results
import ray
import dill
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qpivot,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    plot_circuit,
    Qunitary,
)
import logging
import sys
"""
Instructions to reproduce the ansatz search:

1. Clone hierarqcal and checkout the feature/evolve branch:
   git clone git@github.com:matt-lourens/hierarqcal.git && cd hierarqcal
   git checkout feature/evolve

2. Create venv, activate it, install hierarqcal in editable mode, install extra requirements for this script:
   python3 -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -e .
   cd examples/ansatz_search && pip install -r requirements.txt

3. Run the search. Note the Experiment ID (EXP_ID) from output/logs:
   python main.py

4. Monitor logs. Once 'Best energy mean so far' approaches ~ -0.311, the target ansatz is likely found (can take 15-60+ mins).
   It might also be that you found a very similar ansatz.

5. Analyze the result:
   - Edit analyze_evolution.py, set EXP_ID variable to your run's ID.
   - Run the analysis script cells. The first cell typically plots the best overall ansatz found.

6. Interpreting the Plot / Memory Tables:
   - **Nested Motifs:** If plot_circuit() shows a layer with no gate names, it represents a nested motif. To inspect its structure, assuming 'best_motif' is loaded in analyze_evolution.py, use plot_circuit(best_motif[layer_index].mapping.hierq), replacing 'layer_index' with the index of the layer showing no gate names.
   - **Memory Tables:** The search uses 10 memory tables (indexed 0-9) which are continuously updated. The 'Best energy mean' in logs might refer to a different table than the default one viewed by analyze_evolution.py. To view results from a specific table, change the 'generation' variable in analyze_evolution.py (values 0-9).

The ansatz comes in two equivalent forms usually:
motif = (
    Qcycle(stride=1, offset=0, step=1, mapping=qry, boundary="periodic")
    +Qcycle(mapping=qYZe)    
   
)
plot_circuit(Qinit(6) + motif, plot_width=30)

or

motif = (
    Qcycle(stride=1, offset=0, step=1, mapping=qry, boundary="periodic")
    +Qcycle(mapping=qcry)    
   
)
plot_circuit(Qinit(6) + motif, plot_width=30)

Sometimes the only thing that takes time is removing redundant motifs via dropout, this can always be done in post processing, but it does happen on it's own with some time.

Note: Search time varies due to stochasticity. If the target energy/ansatz isn't found reasonably quickly, you might need to restart main.py.
"""


# update as needed
PATH_EXP = os.path.dirname(__file__)
MODEL = "ising"
CORES = 8
NQ_MAX = 8
OFFSET_MAX = 4
BATCH_SIZE = 7
SAVE_INTERVAL = 10
INIT_POP = 100
P_EXPLORE = .3
GEN_PERIOD = 40
PRESSURE = 0.05
VERBOSE = False
TASK_TIMEOUT = 60 * 10
EXTRAS = BATCH_SIZE
SIZES = [3,4,5]
SIZES_WEIGHTS = [1, 1, 1]
L1 = 1e-4
L2 = 1e-4
MAX_MEM_STORAGE = 10
NUM_TOURNAMENTS = 1
K_RANGE = np.linspace(0, 1, 20)
H_RANGE = np.linspace(0, 1, 20)
HKWARGS = {"h": H_RANGE}
COMMENTS = ""

# fmt: off
hierq_gates = [qx, qy, qz, qrx, qry, qrz, qcrx, qcry, qcrz, qh, qXXe, qYYe, qZZe, qXYe, qXZe, qYXe, qYZe, qZXe, qZYe, qCNOT, qpswap, qiSWAP, qSqrtSwap, q3ent]
# fmt: on

if not os.path.exists(PATH_EXP):
    os.makedirs(PATH_EXP)
experiments_dir = os.path.join(PATH_EXP, "experiments")
if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)
EXP_ID = max([int(d) for d in os.listdir(experiments_dir) if d.isdigit()] + [-1]) + 1
exp_dir = os.path.join(experiments_dir, f"{EXP_ID}")
os.mkdir(exp_dir)
Task_Information = namedtuple(
    "Task_Information", ["motif", "mutation_type", "energy", "nq", "symbols", "fitness"]
)
Task = namedtuple("Task_Information", ["motif", "mutation_type"])
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(exp_dir, "logfile.txt")),
        logging.StreamHandler(sys.stdout),
    ],
)

# fmt: off
logging.info(
    f"===Experiment Setup===\n"
    f"EXP_ID: {EXP_ID}\n"
    f"Model: {MODEL}\n"
    f"CORES: {CORES}\n"
    f"NQ_MAX: {NQ_MAX}\n"
    f"OFFSET_MAX: {OFFSET_MAX}\n"
    f"BATCH_SIZE: {BATCH_SIZE}\n"
    f"SAVE_INTERVAL: {SAVE_INTERVAL}\n"
    f"INIT_POP: {INIT_POP}\n"
    f"P_EXPLORE: {P_EXPLORE}\n"
    f"GEN_PERIOD: {GEN_PERIOD}\n"
    f"PRESSURE: {PRESSURE}\n"
    f"VERBOSE: {VERBOSE}\n"
    f"TASK_TIMEOUT: {TASK_TIMEOUT}\n"
    f"EXTRAS: {EXTRAS}\n"
    f"SIZES: {SIZES}\n"
    f"L1: {L1}\n"
    f"L2: {L2}\n"
    f"MAX_MEM_STORAGE: {MAX_MEM_STORAGE}\n"
    f"NUM_TOURNAMENTS: {NUM_TOURNAMENTS}\n"
    f"H_RANGE: {H_RANGE}\n"
    f"K_RANGE: {K_RANGE}\n"
    f"Gates: {','.join([gate.name for gate in hierq_gates])}\n"
    f"COMMENTS: {COMMENTS}\n"
)
MUTATION_TYPES = {"mu":"mutate", "in":"insert", "lu":"level_up", "cj":"chop_join", "dr":"dropout", "ra":"random"}
# fmt: on
hyperparam_names = [
    "stride",
    "strides",
    "step",
    "steps",
    "offset",
    "offsets",
    "boundary",
    "boundaries",
    "global_pattern",
    "merge_within",
    "share_weights",
    "mapping",
]
get_stride = lambda nq: np.random.choice(range(1, NQ_MAX // 2, 1))
get_step = lambda nq: np.random.choice(range(1, NQ_MAX // 2, 1))
get_offset = lambda nq: np.random.choice(range(4), p=[0.5, 0.4, 0.05, 0.05])
get_strides = lambda nq: [get_stride(nq), get_stride(nq), get_stride(nq)]
get_steps = lambda nq: [get_step(nq), get_step(nq), get_step(nq)]
get_offsets = lambda nq: [get_offset(nq), get_offset(nq), get_offset(nq)]
get_global_pattern = lambda: np.random.choice(
    ["*1", "1*", "*1*", "1*1", "*1*1", "1*1*", "*1*1*", "1*1*1"]
    + ["!0", "0!", "!0!", "0!0", "!0!0", "0!0!", "!0!0!", "0!0!0"]
    + ["*!", "!*", "!*!", "*!*", "01", "10", "101", "010", "1001", "0110"]
)
get_merge_within_pattern = lambda: np.random.choice(["*1", "1*"])
get_boundary = lambda: np.random.choice(["open", "periodic"])
get_boundaries = lambda: [get_boundary(), get_boundary(), get_boundary()]
get_share_weights = lambda: np.random.choice([True, False])
get_mapping = lambda: hierq_gates[np.random.randint(len(hierq_gates))]
get_edge_order = lambda nq: np.random.permutation(range(nq))
Hyperparam_Info = namedtuple("Hyperparam_Info", ["function", "arg_name"])
hyperparam_functions = {
    "stride": Hyperparam_Info(get_stride, "nq"),
    "strides": Hyperparam_Info(get_strides, "nq"),
    "step": Hyperparam_Info(get_step, "nq"),
    "steps": Hyperparam_Info(get_steps, "nq"),
    "offset": Hyperparam_Info(get_offset, "nq"),
    "offsets": Hyperparam_Info(get_offsets, "nq"),
    "boundary": Hyperparam_Info(get_boundary, None),
    "boundaries": Hyperparam_Info(get_boundaries, None),
    "global_pattern": Hyperparam_Info(get_global_pattern, None),
    "merge_within": Hyperparam_Info(get_merge_within_pattern, None),
    "share_weights": Hyperparam_Info(get_share_weights, None),
    "mapping": Hyperparam_Info(get_mapping, None),
    "edge_order": Hyperparam_Info(get_edge_order, "nq"),
}


def reward_function(motif):
    energies = get_results(
        motif, HKWARGS, objective=MODEL, N_iter=50, reps=1, verbose=VERBOSE
    )
    return np.mean(energies)


@ray.remote
def evaluate_task(task, l1=0.00001, l2=0.00001, sizes=[4, 5, 6]):
    motif = task.motif
    energies = []
    for size in sizes:
        hierq = Qinit(tensors=[ket0] * size) + motif
        energies.append(reward_function(hierq) / size)
    # naive gate count
    edges = hierq(get_bits=True)
    if edges is not None:
        # n_gates_arity = len([edge for edge in edges])
        n_gates_arity = sum([len(edge) for edge in edges])
    else:
        n_gates_arity = 0
    n_symbols = hierq.n_symbols
    fitness = (
        (np.array(energies) @ np.array(SIZES_WEIGHTS)) / len(SIZES)
        + n_gates_arity * l1
        + n_symbols * l2
    )
    # fitness = energy5 + n_gates_arity * l1 + n_symbols * l2
    task_info = Task_Information(
        motif,
        task.mutation_type,
        energies,
        sizes,
        [],
        fitness,
    )
    return task_info


def get_random_pivot(mapping=None):
    if mapping == None:
        mapping = hierq_gates[np.random.randint(len(hierq_gates))]
    arity = (
        len(mapping.tail.Q_avail) if isinstance(mapping, Qhierarchy) else mapping.arity
    )
    return Qpivot(
        global_pattern=get_global_pattern(),
        mapping=mapping,
        strides=get_strides(NQ_MAX),
        steps=get_steps(NQ_MAX),
        offsets=get_offsets(NQ_MAX),
        merge_within=get_merge_within_pattern(),
        boundaries=get_boundaries(),
    )


def get_random_cycle(mapping=None):
    if mapping == None:
        mapping = hierq_gates[np.random.randint(len(hierq_gates))]
    return Qcycle(
        mapping=mapping,
        stride=get_stride(NQ_MAX),
        step=get_step(NQ_MAX),
        offset=get_offset(NQ_MAX),
        boundary=get_boundary(),
    )


def get_random_mask(mapping=None):
    if mapping == None:
        # 50/50 chance to have a mapping or not
        mapping = np.random.choice(hierq_gates + [None] * len(hierq_gates))
    if mapping:
        return Qmask(
            global_pattern=get_global_pattern(),
            mapping=mapping,
            strides=get_strides(NQ_MAX),
            steps=get_steps(NQ_MAX),
            offsets=get_offsets(NQ_MAX),
            boundaries=get_boundaries(),
        )
    else:
        return Qmask(global_pattern=get_global_pattern())


def get_random_unmask():
    return Qunmask("previous")


def create_genotype():
    for k in range(5):
        genotype_f = np.random.choice(
            [get_random_cycle, get_random_pivot, get_random_mask, get_random_unmask],
            p=[0.5, 0.3, 0.15, 0.05],
        )
        if isinstance(genotype_f, Qcycle) or isinstance(genotype_f, Qpivot):
            if not (len((Qinit(8) + genotype_f())[1].E) == 0):
                break
        else:
            break
    return genotype_f()


def mutate(motif):
    if isinstance(motif, Qmotifs):
        top_level_indices = range(len(motif))
        selected_index = np.random.choice(top_level_indices)
        selected_motif = motif[selected_index]
    else:
        selected_motif = motif
        selected_index = 0
    hyperparams = {
        hyperparam: vars(selected_motif)[hyperparam]
        for hyperparam in set(vars(selected_motif).keys()) & set(hyperparam_names)
    }
    mutated_param = np.random.choice(list(hyperparams.keys()))
    arg = None
    if hyperparam_functions[mutated_param].arg_name == "nq":
        arg = NQ_MAX
        hyperparams[mutated_param] = hyperparam_functions[mutated_param].function(arg)
    elif hyperparam_functions[mutated_param].arg_name == "arity":
        arg = (
            len(selected_motif.mapping.tail.Q_avail)
            if isinstance(selected_motif.mapping, Qhierarchy)
            else selected_motif.mapping.arity
        )
        hyperparams[mutated_param] = hyperparam_functions[mutated_param].function(arg)
    else:
        hyperparams[mutated_param] = hyperparam_functions[mutated_param].function()

    if isinstance(selected_motif, Qcycle):
        new_motif = Qcycle(**hyperparams)
    elif isinstance(selected_motif, Qpivot):
        new_motif = Qpivot(**hyperparams)
    elif isinstance(selected_motif, Qmask):
        new_motif = Qmask(**hyperparams)
    elif isinstance(selected_motif, Qunmask):
        new_motif = Qunmask(**hyperparams)

    if isinstance(motif, Qmotifs):
        return reduce(
            lambda x, y: x + y,
            [motif[i] if i != selected_index else new_motif for i in range(len(motif))],
        )
    else:
        return new_motif


def mutate_step_up(motif):
    motif_type = np.random.choice(["cycle", "pivot"])
    if motif_type == "cycle":
        new_motif = get_random_cycle(mapping=motif)
    elif motif_type == "pivot":
        new_motif = get_random_pivot(mapping=motif)
    return new_motif


def crossover(motif1, motif2):
    new_motif = motif1 + motif2
    return new_motif


def insert(motif1):
    motif2 = Qmotifs([create_genotype()])
    if isinstance(motif1, Qmotif):
        motif1 = Qmotifs([motif1])
    n1 = len(motif1)
    k1 = np.random.randint(n1)
    new_motif = Qmotifs(motif1[:k1] + motif2 + motif1[k1:])
    return new_motif


def chop_and_join(motif1, motif2):
    if isinstance(motif1, Qmotif):
        motif1 = Qmotifs([motif1])
    if isinstance(motif2, Qmotif):
        motif2 = Qmotifs([motif2])
    n1 = len(motif1)
    n2 = len(motif2)
    k1, k2 = np.random.randint(n1), np.random.randint(n2)
    genotype1 = Qmotifs(motif1[: k1 + 1] + motif2[k2:])
    genotype2 = Qmotifs(motif2[: k2 + 1] + motif1[k1:])
    return genotype1, genotype2


def dropout(motif):
    # this function assumes the provided motif has len>1
    top_level_indices = range(0, len(motif))
    selected_index = np.random.choice(top_level_indices)
    new_motif_tuple = motif[:selected_index] + motif[selected_index + 1 :]
    if len(new_motif_tuple) == 1:
        new_motif = new_motif_tuple[0]
    else:
        new_motif = Qmotifs(motif[:selected_index] + motif[selected_index + 1 :])
    return new_motif


def tournament_selection(memory_table, pressure=0.05, p_explore=0.3):
    num_elements = int(len(memory_table) * pressure)
    if num_elements < 2:
        num_elements = 2

    if np.random.rand() < p_explore:
        selected_tasks_ind = np.random.choice(len(memory_table), 2, replace=False)
        selected_tasks = (
            memory_table[selected_tasks_ind[0]],
            memory_table[selected_tasks_ind[1]],
        )

    else:
        selected_tasks_ind = np.random.choice(
            len(memory_table), num_elements, replace=False
        )
        selected_tasks = sorted(
            [memory_table[ind] for ind in selected_tasks_ind],
            key=lambda x: x.fitness,
        )
    return selected_tasks[0], selected_tasks[1]


def get_initial_population(size=10):
    population = []
    for k in range(size):
        motif = create_genotype()
        population.append(Task(motif, MUTATION_TYPES["ra"]))
    return population


# # === Debug ====
# # with open(f"{PATH}/experiments/{12}/task_table_{1}.pkl", "rb") as file:
# #     task_table = dill.load(file)
# # for task in task_table:
# #     evaluate_task(task)
# import cProfile
# import pstats
# import io
# profiler = cProfile.Profile()
# profiler.enable()
# motif = Qcycle(mapping=qry) + Qcycle(mapping=qcry)
# evaluate_task(motif)
# profiler.disable()
# s = io.StringIO()
# ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
# ps.print_stats(10)
# print(s.getvalue())
# m1 = (
#     Qcycle(mapping=qh)
#     + Qcycle(step=2, mapping=qry)
#     + Qcycle(step=2, offset=1, mapping=qry)
#     + Qcycle(step=2, offset=0, mapping=qcry)
#     + Qcycle(stride=1, step=2, offset=1, mapping=qcry)
# )
# m2 = create_genotype() + create_genotype() + create_genotype() + create_genotype()
# memory_table.extend([evaluate_task(m1), evaluate_task(m2)])
# crossover(memory_table[0], memory_table[1])


# --- The callback function for asynchronous evaluations ---
def generate_offspring(task1, task2):
    cj1, cj2 = chop_and_join(task1.motif, task2.motif)
    inserted_genotype = insert(task1.motif)
    mutated_genotype1 = mutate(task1.motif)
    mutated_genotype2 = mutate(task2.motif)
    random_nq = np.random.randint(2, 5)
    evolved_genotype = mutate_step_up(Qinit(random_nq) + task1.motif)
    if isinstance(task1.motif, Qmotifs):
        if len(task1.motif) > 1:
            dropout_genotype = dropout(task1.motif)
        else:
            dropout_genotype = create_genotype()
    else:
        dropout_genotype = create_genotype()

    return [
        Task(cj1, MUTATION_TYPES["cj"]),
        Task(cj2, MUTATION_TYPES["cj"]),
        Task(inserted_genotype, MUTATION_TYPES["in"]),
        Task(mutated_genotype1, MUTATION_TYPES["mu"]),
        Task(mutated_genotype2, MUTATION_TYPES["mu"]),
        Task(evolved_genotype, MUTATION_TYPES["lu"]),
        Task(dropout_genotype, MUTATION_TYPES["dr"]),
    ]


tournaments = NUM_TOURNAMENTS
if __name__ == "__main__":
    t0 = time.time()
    ray.init(num_cpus=CORES)
    initial_population = get_initial_population(size=INIT_POP)
    memory_table = []
    memory_tables_stored = []
    t0 = time.time()
    results = ray.get(
        [
            evaluate_task.remote(task, l1=L1, l2=L2, sizes=SIZES)
            for task in initial_population
        ]
    )
    memory_table.extend(results)
    t1 = time.time()
    logging.info(f"=== Pool initialisation ===\ntime: {t1-t0}\nsize: {INIT_POP}")
    with open(f"{exp_dir}/memory_table_{0}.pkl", "wb") as f:
        dill.dump(memory_table, f)
    memory_tables_stored.append("memory_table_{0}.pkl")
    n_memory_tables_stored = 1
    last_save_count = INIT_POP
    last_tournament_count = INIT_POP
    task1, task2 = tournament_selection(
        memory_table, pressure=PRESSURE, p_explore=P_EXPLORE
    )
    new_tasks = generate_offspring(task1, task2)
    extra_tasks = get_initial_population(size=EXTRAS)
    unfinished_tasks = [
        evaluate_task.remote(task, l1=L1, l2=L2, sizes=SIZES) for task in new_tasks
    ]
    unfinished_tasks.extend(
        [evaluate_task.remote(task, l1=L1, l2=L2, sizes=SIZES) for task in extra_tasks]
    )
    generation=0
    while True:
        finished_tasks, unfinished_tasks = ray.wait(
            unfinished_tasks, timeout=TASK_TIMEOUT, num_returns=BATCH_SIZE
        )
        memory_table.extend(ray.get(finished_tasks))
        current_count = len(memory_table)
        if len(unfinished_tasks) < BATCH_SIZE:
            tournaments = BATCH_SIZE // 7 + 1
        else:
            tournaments = NUM_TOURNAMENTS
        for tournament in range(tournaments):
            task1, task2 = tournament_selection(
                memory_table, pressure=PRESSURE, p_explore=P_EXPLORE
            ) #(np.cos(2*np.pi/GEN_PERIOD*generation)+1)/2 *np.exp(-1/GEN_PERIOD*generation)
            new_tasks = generate_offspring(task1, task2)
            unfinished_tasks.extend(
                [
                    evaluate_task.remote(task, l1=L1, l2=L2, sizes=SIZES)
                    for task in new_tasks
                ]
            )
        last_tournament_count = current_count
        generation+= 1
        best_item = min(memory_table, key=lambda x: x.fitness)
        best_fitness = best_item.fitness
        energies = best_item.energy
        logging.info(f"=== New Tournament Selection Event ===")
        logging.info(f"Generation: {generation}")
        logging.info(f"Best fitness so far: {best_fitness}")
        logging.info(f"Best energy mean so far: {np.mean(energies)}")
        logging.info(f"Memory table size: {current_count}")
        logging.info(f"Unfinished: {len(unfinished_tasks)}")
        logging.info(f"Time elapsed: {time.time() - t0}")
        logging.info(f"P_explore: {P_EXPLORE}")
        if current_count - last_save_count >= SAVE_INTERVAL:
            with open(
                f"{exp_dir}/memory_table_{n_memory_tables_stored}.pkl",
                "wb",
            ) as f:
                dill.dump(memory_table, f)
            last_save_count = current_count
            n_memory_tables_stored += 1
            n_memory_tables_stored = (
                0
                if n_memory_tables_stored > MAX_MEM_STORAGE
                else n_memory_tables_stored
            )
