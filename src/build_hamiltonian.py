"""
build_hamiltonian.py
--------------------
Build the second-quantized and qubit Hamiltonian for LiH at a given bond length
using PySCF as the backend driver and Qiskit Nature for the mapping.

The frozen-core approximation is applied to reduce the number of qubits required
by removing the Li 1s core orbital and the corresponding virtual.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


def build_hamiltonian(bond_length: float):
    """Build the LiH Hamiltonian at the specified bond length.

    Parameters
    ----------
    bond_length : float
        Li–H internuclear distance in Angstroms.

    Returns
    -------
    qubit_op : SparsePauliOp
        The qubit Hamiltonian obtained via the parity mapping (with two-qubit
        reduction enabled).
    problem : ElectronicStructureProblem
        The reduced electronic-structure problem (frozen core applied).
    mapper : ParityMapper
        The mapper used so that downstream code can reuse the same mapping
        when needed (e.g. for initial-state preparation).
    """
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    problem = driver.run()

    # Freeze the Li 1s core orbital to reduce qubit count
    transformer = FreezeCoreTransformer()
    problem = transformer.transform(problem)

    # Parity mapping with two-qubit reduction takes advantage of Z2 symmetries
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])

    return qubit_op, problem, mapper
