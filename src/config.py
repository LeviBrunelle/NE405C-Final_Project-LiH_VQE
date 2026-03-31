from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RUN_NAME = "slide-replacement"
EXPERIMENT = "optimizer"  # single, reps, optimizer, ansatz, repeat_single, 
                       # repeat_optimizer, repeat_ansatz, bond_scan, basis,
                       # optimizer_reps, noise

MOLECULE_NAME = "LiH"  # LiH or BeH2
BASIS = "sto3g"
BASIS_SETS = ["sto3g", "3-21g"]
CHARGE = 0
SPIN = 0

DEFAULT_BOND_LENGTH = 0.7
BOND_LENGTHS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MAX_EXACT_QUBITS = 16

ANSATZ_KIND = "hardware_efficient"  # hardware_efficient or uccsd_hf
ANSATZ_KINDS = ["hardware_efficient", "uccsd_hf"]
ENTANGLEMENT_PATTERN = "linear"
REPS = 2
REPS_VALUES = [1, 2, 3]

OPTIMIZER = "COBYLA"  # COBYLA, SPSA
OPTIMIZERS = ["COBYLA", "SPSA"]
COBYLA_MAXITER = 4000
SPSA_MAXITER = COBYLA_MAXITER // 2
GD_MAXITER = 12
GD_LEARNING_RATE = 0.05

BACKEND_MODE = "statevector"  # statevector, aer_shots, aer_noise
BACKEND_MODES = ["statevector", "aer_shots", "aer_noise"]
AER_DEFAULT_PRECISION = 5e-2
NOISE_1Q = 0.001
NOISE_2Q = 0.01
READOUT_ERROR = 0.02

NUM_TRIALS = 5
RANDOM_SEED = 123
USE_TQDM = True

SAVE_PLOTS = True
SAVE_SUMMARIES = True
SAVE_ANSATZ_FIGURE = True
FIG_DPI = 300
ANSATZ_FOLD = 50
PREVIEW_TERM_COUNT = 15


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
