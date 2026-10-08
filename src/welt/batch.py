"""Immutable terminal batch manifests and explicit, validated numerical payloads."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from types import MappingProxyType

from .errors import InvalidBatchResultError

TERMINAL = ('succeeded', 'partially_completed', 'failed', 'cancelled')
PROTOCOL = 'welt-batch-contiguous-v1'


def invalid():
    raise InvalidBatchResultError('Batch response does not match its owned pinned result.', code='invalid_batch_result')


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def text(value):
    return isinstance(value, str) and 1 <= len(value) <= 128


def finite(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def scalar(value):
    return isinstance(value, (str, bool)) or finite(value)


def job_binding(job, job_id, *, terminal=True):
    if (not isinstance(job, dict) or job.get('id') != str(job_id)
            or job.get('operation') != 'batch_predict'
            or job.get('task') not in ('classification', 'regression')
            or not text(job.get('model_version')) or not text(job.get('predictor_id'))
            or not text(job.get('operation_id'))):
        invalid()
    if terminal and job.get('status') not in TERMINAL:
        from .errors import ConflictError
        raise ConflictError('Batch result is not ready.', code='result_not_ready', job_id=str(job_id), retryable=True)
    return job


def checked_ranges(successful, failed, rows):
    if not isinstance(successful, (list, tuple)) or not isinstance(failed, (list, tuple)):
        invalid()
    ranges = []
    for entries, outcome in ((successful, 'success'), (failed, 'failure')):
        previous = -1
        for bounds in entries:
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(type(value) is not int for value in bounds)):
                invalid()
            start, stop = bounds
            if not 0 <= start < stop <= rows or stop - start > 100 or start < previous:
                invalid()
            previous = stop
            ranges.append((start, stop, outcome))
    end = 0
    for start, stop, _ in sorted(ranges):
        if start != end:
            invalid()
        end = stop
    if end != rows:
        invalid()
    return ranges


@dataclass(frozen=True, init=False)
class BatchResult:
    _data: object
    status: str

    def __init__(self, value, job):
        fields = {'job_id','row_count','successful_ranges','failed_ranges','result_reference','expires_at','model_version'}
        if not isinstance(value, dict) or set(value) != fields:
            invalid()
        job_binding(job, value.get('job_id'))
        if (value['job_id'] != job['id'] or value['model_version'] != job['model_version']
                or type(value['row_count']) is not int or not 1 <= value['row_count'] <= 10000
                or not text(value['result_reference'])):
            invalid()
        try:
            when = datetime.fromisoformat(value['expires_at'])
            if when.tzinfo is None:
                invalid()
        except (TypeError, ValueError, OverflowError):
            invalid()
        checked_ranges(value['successful_ranges'],value['failed_ranges'],value['row_count'])
        if (job['status'] == 'succeeded' and value['failed_ranges'] or
            job['status'] == 'partially_completed' and (not value['failed_ranges'] or not value['successful_ranges']) or
            job['status'] == 'failed' and value['successful_ranges']):
            invalid()
        object.__setattr__(self,'_data',freeze(deepcopy(value)))
        object.__setattr__(self,'status',job['status'])

    def to_dict(self):
        return thaw(self._data)

    @property
    def job_id(self): return self._data['job_id']
    @property
    def row_count(self): return self._data['row_count']
    @property
    def successful_ranges(self): return self._data['successful_ranges']
    @property
    def failed_ranges(self): return self._data['failed_ranges']
    @property
    def result_reference(self): return self._data['result_reference']
    @property
    def expires_at(self): return self._data['expires_at']
    @property
    def model_version(self): return self._data['model_version']


@dataclass(frozen=True, init=False)
class BatchPayload:
    _data: object

    def __init__(self, value, result, job):
        fields={'job_id','result_reference','row_count','model_version','predictor_id','operation_id',
                'batch_protocol','predictions','probabilities','classes','errors'}
        if not isinstance(value,dict) or set(value)!=fields:
            invalid()
        job_binding(job,result.job_id)
        for key in ('job_id','result_reference','row_count','model_version'):
            if type(value[key]) is not type(result.to_dict()[key]) or value[key] != result.to_dict()[key]:
                invalid()
        if (value['predictor_id'] != job['predictor_id'] or value['operation_id'] != job['operation_id']
                or value['batch_protocol'] != PROTOCOL):
            invalid()
        rows=result.row_count
        predictions=value['predictions'];vectors=value['probabilities'];labels=value['classes']
        if not isinstance(predictions,(list,tuple)) or len(predictions)!=rows:
            invalid()
        if labels is not None:
            if (not isinstance(labels,(list,tuple)) or not 2<=len(labels)<=10
                    or any(not scalar(label) for label in labels)
                    or len({(type(label),label) for label in labels}) != len(labels)):
                invalid()
        if job.get('task') not in ('classification','regression'):
            invalid()
        if (job['task']=='regression') != (labels is None):
            invalid()
        if vectors is not None and (labels is None or not isinstance(vectors,(list,tuple)) or len(vectors)!=rows):
            invalid()
        if not isinstance(value['errors'],(list,tuple)) or len(value['errors']) != len(result.failed_ranges):
            invalid()
        for error,bounds in zip(value['errors'],result.failed_ranges):
            if (not isinstance(error,dict) or set(error) != {'range','code','retryable','outcome'}
                or error['range'] != list(bounds) or type(error['retryable']) is not bool
                or error['code'] not in ('execution_failed','execution_interrupted','worker_unavailable',
                                        'version_unavailable','cancelled','execution_timeout')
                or error['outcome'] not in ('failed','cancelled')
                or ((error['outcome']=='cancelled') != (error['code']=='cancelled'))
                or error['code']=='cancelled' and error['retryable']):
                invalid()
        failed={i for start,stop in result.failed_ranges for i in range(start,stop)}
        for index,prediction in enumerate(predictions):
            if index in failed:
                if prediction is not None or vectors is not None and vectors[index] is not None:
                    invalid()
                continue
            if labels is None:
                if not finite(prediction): invalid()
            elif not any(type(prediction) is type(label) and prediction==label for label in labels):
                invalid()
            if vectors is not None:
                vector=vectors[index]
                if (not isinstance(vector,(list,tuple)) or len(vector)!=len(labels)
                    or any(not finite(v) or not 0<=v<=1 for v in vector)
                    or not math.isclose(sum(vector),1,abs_tol=1e-5,rel_tol=1e-5)):
                    invalid()
        object.__setattr__(self,'_data',freeze(deepcopy(value)))

    def to_dict(self): return thaw(self._data)
    @property
    def job_id(self): return self._data['job_id']
    @property
    def predictions(self): return self._data['predictions']
    @property
    def probabilities(self): return self._data['probabilities']
    @property
    def classes(self): return self._data['classes']
    @property
    def errors(self): return self._data['errors']
