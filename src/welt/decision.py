"""Decision requests and ranked results, with no local causal fitting."""
from copy import deepcopy
from dataclasses import dataclass
import math

from .client import Job, AsyncJob


def _source(value):
    if isinstance(value, str):
        return {'dataset_id': value}
    if isinstance(value, dict):
        if 'dataset_id' in value or 'table' in value:
            return deepcopy(value)
        if 'columns' in value and 'rows' in value:
            return {'table': deepcopy(value)}
    if hasattr(value, 'columns') and hasattr(value, 'itertuples'):
        import pandas as pd
        def cell(v):
            if pd.isna(v):
                return None
            return v.item() if hasattr(v, 'item') else v
        return {'table': {'columns': list(value.columns),
                          'rows': [[cell(v) for v in row] for row in value.itertuples(index=False, name=None)]}}
    raise TypeError('Use a DataFrame, dataset ID, or a named table for context and rows.')


def decision_request(context, *, target, target_kind, discovery_model, estimation_model,
                     interventions, rows=None, objective='maximize', target_class=None,
                     knowledge=None, types=None, n_folds=5, seed=0, min_propensity=.01, max_fits=500):
    """Create the wire request; methods and explicit/default contrasts belong to each variable."""
    if objective not in ('maximize','minimize'):
        raise ValueError('objective must be maximize or minimize')
    if target_kind not in ('continuous','binary','categorical'):
        raise ValueError('target_kind must be continuous, binary or categorical')
    if not interventions or target in interventions:
        raise ValueError('List actionable variables other than the target.')
    return dict(context=_source(context), rows=None if rows is None else _source(rows), target=target,
                target_kind=target_kind, target_class=target_class, objective=objective,
                discovery_model=discovery_model, estimation_model=estimation_model,
                interventions=deepcopy(interventions), knowledge=deepcopy(knowledge or {}),
                types=dict(types or {}), n_folds=n_folds, seed=seed,
                min_propensity=min_propensity, max_fits=max_fits)


@dataclass(frozen=True)
class DecisionResult:
    """Ranked population actions and query-row conditional effects with provenance.

    Access ``population``, ``individual`` and ``skipped`` directly, or ``to_dict``
    for the complete graph, model versions, diagnostics and uncertainty semantics.
    """
    _data: dict

    @classmethod
    def from_dict(cls, data):
        if (not isinstance(data, dict) or data.get('task') != 'decision'
                or data.get('protocol') != 'decision-contrasts_v1'):
            raise ValueError('Unsupported decision result contract.')
        for name in ('population','individual','skipped'):
            if not isinstance(data.get(name), list):
                raise ValueError(f'Decision result is missing {name}.')
        groups = [data['population']] + [row['interventions'] for row in data['individual']]
        for group in groups:
            for rank, item in enumerate(group, 1):
                if item.get('rank') != rank or not math.isfinite(item['effect']):
                    raise ValueError('Decision ranking or effect is invalid.')
        return cls(deepcopy(data))

    @property
    def population(self):
        return deepcopy(self._data['population'])

    @property
    def individual(self):
        return deepcopy(self._data['individual'])

    @property
    def skipped(self):
        return deepcopy(self._data['skipped'])

    def to_dict(self):
        return deepcopy(self._data)


class DecisionJob(Job):
    """Durable decision handle; timeout does not cancel the server job."""
    def result(self, *, timeout=600, poll_interval=.25):
        return DecisionResult.from_dict(super().result(timeout=timeout, poll_interval=poll_interval))


class AsyncDecisionJob(AsyncJob):
    """Async durable decision handle with the same result semantics."""
    async def result(self, *, timeout=600, poll_interval=.25):
        return DecisionResult.from_dict(await super().result(timeout=timeout, poll_interval=poll_interval))


class Decisions:
    """Synchronous ``client.decisions`` resource; create waits, submit returns a job."""
    def __init__(self, client):
        self.client = client

    def submit(self, context, *, idempotency_key=None, **kwargs):
        return self.client.submit_decision(decision_request(context, **kwargs), idempotency_key=idempotency_key)

    def create(self, context, *, timeout=600, poll_interval=.25, idempotency_key=None, **kwargs):
        return self.submit(context, idempotency_key=idempotency_key, **kwargs).result(
            timeout=timeout, poll_interval=poll_interval)

    def job(self, job_id):
        return DecisionJob(self.client, job_id)

    def delete(self, job_id):
        return self.client.delete_decision(job_id)


class AsyncDecisions:
    """Async ``client.decisions`` resource; submission and waiting remain separate."""
    def __init__(self, client):
        self.client = client

    async def submit(self, context, *, idempotency_key=None, **kwargs):
        return await self.client.submit_decision(decision_request(context, **kwargs), idempotency_key=idempotency_key)

    async def create(self, context, *, timeout=600, poll_interval=.25, idempotency_key=None, **kwargs):
        job = await self.submit(context, idempotency_key=idempotency_key, **kwargs)
        return await job.result(timeout=timeout, poll_interval=poll_interval)

    def job(self, job_id):
        return AsyncDecisionJob(self.client, job_id)

    async def delete(self, job_id):
        return await self.client.delete_decision(job_id)
