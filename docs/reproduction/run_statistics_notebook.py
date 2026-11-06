"""Execute the frozen statistics notebook into a new file, preserving the original."""
import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Extracted report-analysis/notebook/data directory')
    parser.add_argument('--out', type=Path, required=True, help='New output notebook')
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if not data.is_dir() or out.exists():
        parser.error('Data must exist and the output file must be new.')
    source = Path(__file__).resolve().parents[2] / 'v2_sensors/driving_v2/report_statistics.ipynb'
    notebook = nbformat.read(source, as_version=4)
    os.environ['COMP0245_REPORT_DATA'] = str(data)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    NotebookClient(notebook, km=manager, timeout=300, allow_errors=False).execute()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        nbformat.write(notebook, handle)
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'PASS: executed all {count} code cells; saved {out}')


if __name__ == '__main__':
    main()
