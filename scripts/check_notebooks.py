"""Validate scrubbed sources and execute in disposable Jupyter kernels, never saving outputs."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true')
    parser.add_argument("--check-source", action="store_true", help="Validate scrubbed standalone sources without claiming execution.")
    paths=sorted((ROOT/'examples/notebooks').glob('*.ipynb'))
    parser.add_argument('--notebook',choices=[p.stem for p in paths],help='Execute one named example; default executes all.')
    args=parser.parse_args()
    env=os.environ.copy()
    env['WELT_NOTEBOOK_MODE']='live' if args.live else 'fixture'
    if not args.live:
        env.pop('WELT_API_KEY',None);env.pop('WELT_BASE_URL',None)
    import welt
    if 'site-packages' not in str(Path(welt.__file__)):
        raise SystemExit('Install the built SDK wheel into the execution environment first.')
    for path in paths:
        if args.notebook and path.stem!=args.notebook:continue
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code' and (cell.outputs or cell.execution_count is not None):
                raise SystemExit(f'Notebook source must be scrubbed: {path.name}')
        with tempfile.TemporaryDirectory(prefix='welt-notebook-') as temporary:
            folder=Path(temporary)
            if args.check_source:
                print(f"PASS {path.name} (source validation only; not execution)")
                continue
            if not args.live:
                fixture_dir = os.environ.get("WELT_NOTEBOOK_FIXTURE_DIR")
                if not fixture_dir:
                    raise SystemExit("Offline execution requires WELT_NOTEBOOK_FIXTURE_DIR with reviewed SHA-pinned research bytes; use --check-source for source checks.")
                shutil.copy(ROOT/'scripts/notebook_fixture.py', folder/'notebook_fixture.py')
                setup = nbformat.v4.new_code_cell(
                    "from notebook_fixture import patch_clients\n"
                    "import os\n_fixture = patch_clients(os.environ['WELT_NOTEBOOK_FIXTURE_DIR'])")
                notebook.cells.insert(0, setup)
            kernel_folder=folder/'kernels'/'welt-examples';kernel_folder.mkdir(parents=True)
            (kernel_folder/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Welt examples',language='python')))
            manager=KernelManager(kernel_name='welt-examples',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(folder/'kernels')]))
            try:
                NotebookClient(notebook,km=manager,timeout=150,startup_timeout=60,
                    store_widget_state=False,record_timing=False).execute(cwd=str(folder),env=env,cleanup_kc=True)
            except Exception:
                # CellExecutionError can contain values. Keep diagnostics bounded and private.
                raise SystemExit(f'Notebook failed: {path.name}; inspect locally without committing outputs.') from None
        print(f'PASS {path.name} ({"live" if args.live else "test transport; real pinned research input; no model quality evidence"}); outputs not saved')


if __name__=='__main__':main()
