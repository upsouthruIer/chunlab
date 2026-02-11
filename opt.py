import os
import glob
import json
import numpy as np

from ase.io import read, write
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip


def make_calc(model_name: str = "uma-s-1p1", task_name: str = "oc20"):
    """Create ML calculator (fairchem) for ASE."""
    predictor = pretrained_mlip.get_predict_unit(model_name)
    return FAIRChemCalculator(predictor, task_name=task_name)


def run_batch_optimizations(    
    input_dir: str,
    pattern: str = "POSCAR_*",
    model_name: str = "uma-s-1p1",
    task_name: str = "oc20",
    fmax: float = 0.05,
    steps: int = 300,
    output_dir: str | None = None,
    save_trajectory: bool = False,
):
    """
    Batch-optimize all POSCAR-like files in input_dir matching pattern.
    Saves:
      - logs: opt_<filename>.log
      - optimized structures: <filename>_relaxed.vasp (or .traj optional)
      - energies: energies.json + energies.csv
    """
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    if output_dir is None:
        output_dir = os.path.join(input_dir, "relaxed_results")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found: {os.path.join(input_dir, pattern)}")

    calc = make_calc(model_name=model_name, task_name=task_name)

    energies = {}      # {basename: energy}
    failures = {}      # {basename: error_message}

    for path in files:
        base = os.path.basename(path)

        log_path = os.path.join(output_dir, f"opt_{base}.log")
        traj_path = os.path.join(output_dir, f"{base}.traj")
        out_vasp = os.path.join(output_dir, f"{base}_relaxed.vasp")

        try:
            atoms = read(path)
            atoms.set_calculator(calc)

            # Optimizer
            if save_trajectory:
                opt = BFGS(atoms, logfile=log_path, trajectory=traj_path)
            else:
                opt = BFGS(atoms, logfile=log_path)

            opt.run(fmax=fmax, steps=steps)

            e = atoms.get_potential_energy()
            energies[base] = float(e)

            # Save relaxed structure (VASP)
            # vasp5=True, direct=True: usual POSCAR style
            write(out_vasp, atoms, format="vasp", vasp5=True, direct=True)

            print(f"[OK] {base:30s}  E = {e:.6f} eV  -> {os.path.basename(out_vasp)}")

        except Exception as exc:
            failures[base] = repr(exc)
            print(f"[FAIL] {base}: {exc}")

    # Save energies (JSON + CSV)
    json_path = os.path.join(output_dir, "energies.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(energies, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "energies.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,energy_eV\n")
        for k in sorted(energies.keys()):
            f.write(f"{k},{energies[k]:.12f}\n")

    # Save failures if any
    if failures:
        fail_path = os.path.join(output_dir, "failures.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Total: {len(files)} | Success: {len(energies)} | Fail: {len(failures)}")
    print(f"Results saved in: {output_dir}")
    print(f"- {os.path.basename(json_path)}")
    print(f"- {os.path.basename(csv_path)}")
    if failures:
        print(f"- failures.json (errors)")


"""if __name__ == "__main__":
    run_batch_optimizations(
        input_dir="/Users/seungsoohan/chun_lab/3rd_trial/31",
        pattern="POSCAR_*",
        model_name="uma-s-1p1",
        task_name="oc20",
        fmax=0.05,
        steps=300,
        output_dir="/Users/seungsoohan/chun_lab/3rd_trial/11/relaxed_results",
        save_trajectory=False,  # True로 하면 각 구조 최적화 궤적(.traj)도 저장
    )"""



if __name__ == "__main__":
    run_batch_optimizations(
        input_dir=".",              # ← 현재 디렉토리
        pattern="POSCAR_*",
        model_name="uma-s-1p1",
        task_name="oc20",
        fmax=0.05,
        steps=300,
        output_dir="./relaxed_results",  # ← 현재 디렉토리 하위에 결과 저장
        save_trajectory=False,
    )
