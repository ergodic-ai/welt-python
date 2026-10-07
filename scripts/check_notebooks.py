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
    args=parser.parse_args()
    env=os.environ.copy()
    env['WELT_NOTEBOOK_MODE']='live' if args.live else 'fixture'
    if not args.live:
        env.pop('WELT_API_KEY',None);env.pop('WELT_BASE_URL',None)
    import welt
    if 'site-packages' not in str(Path(welt.__file__)):
        raise SystemExit('Install the built SDK wheel into the execution environment first.')
    for path in sorted((ROOT/'examples/notebooks').glob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code' and (cell.outputs or cell.execution_count is not None):
                raise SystemExit(f'Notebook source must be scrubbed: {path.name}')
        with tempfile.TemporaryDirectory(prefix='welt-notebook-') as temporary:
            folder=Path(temporary);shutil.copy(ROOT/'examples/notebooks/support.py',folder/'support.py')
            kernel_folder=folder/'kernels'/'welt-examples';kernel_folder.mkdir(parents=True)
            (kernel_folder/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Welt examples',language='python')))
            manager=KernelManager(kernel_name='welt-examples',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(folder/'kernels')]))
            try:
                NotebookClient(notebook,km=manager,timeout=150,startup_timeout=60,
                    store_widget_state=False,record_timing=False).execute(cwd=str(folder),env=env,cleanup_kc=True)
            except Exception:
                # CellExecutionError can contain values. Keep diagnostics bounded and private.
                raise SystemExit(f'Notebook failed: {path.name}; inspect locally without committing outputs.') from None
        print(f'PASS {path.name} ({"live" if args.live else "controlled synthetic transport"}); outputs not saved')


if __name__=='__main__':main()
