"""Prevent docs from falsely publishing a mismatched/incomplete installed SDK reference."""
import sys
import importlib.util

import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_docs, check_notebooks


class ReferenceGates(unittest.TestCase):
    def test_stale_installed_package_cannot_publish_current_reference(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sys,'argv',['build_docs','--output',folder]), patch.object(build_docs,'installed_version',return_value='0.0.0'):
            with self.assertRaisesRegex(SystemExit,'version must match'):
                build_docs.main()
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_missing_async_interface_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sys,'argv',['build_docs','--output',folder]):
            with patch.object(build_docs.welt,'AsyncClient'):
                original=build_docs.welt.AsyncClient
                del build_docs.welt.AsyncClient
                try:
                    with self.assertRaisesRegex(SystemExit,'required documented public symbol'):
                        build_docs.main()
                finally:build_docs.welt.AsyncClient=original
            self.assertEqual(list(Path(folder).iterdir()),[])


class NotebookSelection(unittest.TestCase):
    def test_unknown_selection_rejects_before_kernel_execution(self):
        with patch.object(sys,'argv',['check_notebooks','--notebook','not-available']), patch.object(check_notebooks,'NotebookClient') as execute:
            with self.assertRaises(SystemExit):check_notebooks.main()
            execute.assert_not_called()


class RegressionNotebookFixture(unittest.TestCase):
    @staticmethod
    def support():
        path=Path(__file__).resolve().parents[1]/'examples/notebooks/support.py'
        spec=importlib.util.spec_from_file_location('notebook_support',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_regression_points_reorder_reopen_and_task_identity(self):
        support=self.support()
        with patch.dict('os.environ',{'WELT_NOTEBOOK_MODE':'fixture','WELT_API_KEY':'real-key-must-not-be-read'}):
            with support.notebook_client() as client:
                columns,query,predictor=support.prepare_regression(client)
                result=client.predict(predictor['id'],columns=columns,rows=query.tolist())
                reordered=client.predict(predictor['id'],columns=columns[::-1],rows=query[:,::-1].tolist())
                self.assertEqual(predictor['task'],'regression')
                self.assertEqual(predictor['model_version'],'synthetic-regression-contract-v1')
                self.assertIsNone(result['classes']);self.assertIsNone(result['probabilities'])
                self.assertEqual(np.shape(result['predictions']),(32,))
                self.assertTrue(np.isfinite(result['predictions']).all())
                np.testing.assert_allclose(result['predictions'],reordered['predictions'])
                self.assertEqual(client.predictor(predictor['id'])['model_version'],predictor['model_version'])
                self.assertTrue(all(e['task']=='regression' for e in client.usage()))

    def test_missing_regression_profile_rejects_before_upload(self):
        support=self.support()
        from unittest.mock import Mock
        client=Mock();client.models.return_value=[dict(id='tabicl-v2',status='available',tasks=['classification'])]
        with self.assertRaisesRegex(RuntimeError,'Qualified regression worker unavailable'):
            support.prepare_regression(client)
        client.upload.assert_not_called();client.submit_fit.assert_not_called()

    def test_regression_rng_matches_frozen_planted_fixture(self):
        support=self.support();columns,train,target,query=support.regression_table()
        rng=np.random.default_rng(42);expected_train=rng.normal(size=(128,4));expected_query=rng.normal(size=(32,4))
        weights=rng.normal(size=4);expected_target=expected_train@weights+rng.normal(scale=0.05,size=128)
        self.assertEqual(len(columns),4)
        np.testing.assert_array_equal(train,expected_train);np.testing.assert_array_equal(query,expected_query)
        np.testing.assert_array_equal(target,expected_target)


class FileNotebookFixture(unittest.TestCase):
    @staticmethod
    def support():
        return RegressionNotebookFixture.support()

    def test_lost_ack_resume_same_identity_schema_and_async(self):
        from welt import TransportError
        support=self.support()
        import asyncio
        schema=dict(columns=['code','value','flag','target'],types=['string','number','boolean','number'])
        body=b'code,value,flag,target\n0012,1.5,true,0\n0042,,false,1\n'
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{'WELT_NOTEBOOK_MODE':'fixture','WELT_API_KEY':'must-not-be-read'}):
            path=Path(folder)/'data.csv';path.write_bytes(body)
            service=support.UploadSyntheticService()
            options=support.notebook_options()
            import httpx
            options['transport']=httpx.MockTransport(service)
            with support.Client(**options) as client:
                with self.assertRaises(TransportError) as raised:
                    client.upload_file(path,target='target',schema=schema)
                uid=raised.exception.upload_id
                self.assertEqual(client.upload_info(uid)['received_indices'],[0])
                result=client.resume_upload(uid,path,target='target',schema=schema)
                self.assertEqual(result['types'],['categorical','numeric','categorical','numeric'])
                self.assertEqual(result['rows'],2)
                self.assertEqual(service.file_tables[result['id']],[['0012',1.5,True,0.0],['0042',None,False,1.0]])
                self.assertEqual(set(result),{'id','name','rows','columns','types','target','created_at'})
                self.assertEqual(client.resume_upload(uid,path,target='target',schema=schema)['id'],result['id'])
                self.assertEqual(client.upload_info(uid)['status'],'complete')
                self.assertEqual(len(client.datasets()),1)
                self.assertEqual(client.dataset(result['id'])['columns'],schema['columns'])
            async def check():
                async with support.notebook_async_file_client() as client:
                    result=await client.upload_file(path,target='target',schema=schema)
                    self.assertEqual(result['rows'],2)
                    self.assertEqual(result['types'],['categorical','numeric','categorical','numeric'])
            asyncio.run(check())


if __name__=='__main__':unittest.main()
