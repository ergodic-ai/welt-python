"""Execute published teaching programs with test-only patching and pinned research files.

Fixture prediction metrics are suppressed and never counted as model evidence.
This optional local release check requires no hosted request or customer credential.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

from notebook_fixture import patch_clients

ROOT = Path(__file__).resolve().parents[1]


def main():
    import os
    directory = os.environ.get('WELT_NOTEBOOK_FIXTURE_DIR')
    if not directory:
        raise SystemExit('Set WELT_NOTEBOOK_FIXTURE_DIR to reviewed canonical research fixtures.')
    os.environ['WELT_API_KEY'] = 'test-only-key'
    start = (ROOT/'docs/start.md').read_text()
    programs = re.findall(r'```python\n(.*?)```', start, re.S)[1:]
    receipt = []
    for task, program in zip(('classification', 'regression'), programs):
        for run in range(2):  # Repeated public execution gets a fresh local path.
            with patch_clients(directory), redirect_stdout(io.StringIO()):
                namespace = {}
                exec(program, namespace)
                expected = (120, 30, 4) if task == 'classification' else (246, 62, 6)
                train, test, query = (namespace[x] for x in ('train','test','query'))
                assert len(train) == expected[0] and len(test) == expected[1]
                assert query.shape == expected[1:]
                assert set(train.index).isdisjoint(test.index)
                assert len(train) + len(test) == len(namespace['data'])
                assert namespace['target'] not in query.columns
                assert namespace['predictions'].shape == (expected[1],)
                receipt.append((task, run, expected))
    # Resource guides are explicit continuations of the real Iris walkthrough.
    for name in ('async-and-errors','reuse','usage','model-selection','causal-discovery','research-datasets','migration'):
        with patch_clients(directory), redirect_stdout(io.StringIO()):
            namespace={};exec(programs[0],namespace)
            blocks=re.findall(r'```python\n(.*?)```',(ROOT/'docs'/f'{name}.md').read_text(),re.S)
            for block in blocks:
                if 'your-recorded-job-id' in block:
                    compile(block, name, 'exec')  # Recorded-identity recovery template.
                    continue
                exec(block, namespace)
        receipt.append((name, 'continuation', 'PASS'))
    for name, program in (("regression", programs[1]), ("own-data", programs[0])):
        with patch_clients(directory), redirect_stdout(io.StringIO()):
            namespace = {}; exec(program, namespace)
            if name == "own-data":
                namespace["data"] = namespace["data"].rename(columns={namespace["target"]: "churn"})
            for block in re.findall(r'```python\n(.*?)```', (ROOT/'docs'/f'{name}.md').read_text(), re.S):
                exec(block, namespace)
        receipt.append((name, 'continuation', 'PASS'))
    # The scripts share the same inputs, holdout and metric paths as the docs.
    import importlib.util
    for name in ('first_prediction','first_request'):
        path=ROOT/'examples'/f'{name}.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch_clients(directory), redirect_stdout(io.StringIO()), patch.dict(os.environ, {'WELT_API_KEY':'test-only-key'}):
            if name=='first_prediction':
                for task in ('classification','regression'):module.run(task)
            else:
                with tempfile.TemporaryDirectory() as folder, patch('sys.argv',[name,'--output',str(Path(folder)/'iris.parquet')]):
                    module.main()
                    module.main()  # Exact SHA-checked cache reuse without overwrite.
        receipt.append((name, 'script', 'PASS'))
    print(json.dumps({'scope':'test transport; actual research inputs; no model-quality evidence', 'checks':receipt},indent=2))


if __name__ == '__main__':main()
