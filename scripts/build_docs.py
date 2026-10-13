"""Build a local, versioned SDK documentation site from the matching installed wheel.

API-068 / OBJ-002 / MS-010. Historical pages are rebuilt separately from their
immutable release source; this builder never relabels an older SDK contract.
"""
from pathlib import Path
import argparse
import hashlib
import html
import inspect
from importlib.metadata import version as installed_version
import json
import os
import subprocess
import re
import secrets
import shutil
import tomllib
from urllib.parse import quote

import markdown
import welt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "https://ergodic-ai.github.io/welt-python"
REQUIRED = ('Client', 'Job', 'Classifier', 'Regressor', 'AsyncClient', 'AsyncJob',
            'CausalDiscovery', 'CausalResult', 'Forecast', 'ForecastResult', 'ForecastJob', 'AsyncForecastJob')
# DEC-073: top sections and the guide sidebar. Order defines the previous/next pager.
NAV = [('Get started', 'start'), ('Guides', 'own-data'), ('Reference', 'reference'), ('Models', 'capabilities'), ('CLI', 'cli')]
GUIDES = [('Get started', [('First prediction', 'start'), ('Connect your workspace', 'connect'), ('Install and accounts', 'onboarding'),
                           ('Notebook examples', 'notebooks')]),
          ('Guides', [('Use your own table', 'own-data'), ('Predict a number', 'regression'), ('Reuse a predictor', 'reuse'),
                      ('Select a model', 'model-selection'), ('Forecast a time series', 'forecasting'), ('Hidden model runs', 'model-runs'),
                      ('Research datasets', 'research-datasets'), ('Wait and recover', 'async-and-errors'),
                      ('Discover a graph', 'causal-discovery'), ('Workspace resources', 'usage')]),
          ('Reference', [('Python reference', 'reference'), ('Models and limits', 'capabilities'), ('Agent CLI', 'cli')]),
          ('Releases', [('Upgrade your client', 'migration')])]
SECTION = {'index': 'Get started', 'start': 'Get started', 'connect': 'Get started', 'onboarding': 'Get started',
           'notebooks': 'Get started', 'reference': 'Reference', 'capabilities': 'Models', 'cli': 'CLI'}
TASKS = [('Classify rows', 'start', 'Predict a category for each row of a table.'),
         ('Predict a number', 'regression', "Regression that returns values in your target's units."),
         ('Use your own table', 'own-data', 'DataFrames, CSV and Parquet with a named target.'),
         ('Discover a causal graph', 'causal-discovery', 'Native graphs from observational data.'),
         ('Research datasets', 'research-datasets', 'Search, inspect and load real tables to practise on.'),
         ('Reuse a predictor', 'reuse', 'Predict again later without another fit.')]
REPOSITORY = 'https://github.com/ergodic-ai/welt-python'
ICONS = {
    'menu': '<path d="M4 6h16M4 12h16M4 18h16"/>',
    'search': '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    'theme': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
    'chevron': '<path d="m9 18 6-6-6-6"/>',
    'left': '<path d="M19 12H5M11 18l-6-6 6-6"/>',
    'right': '<path d="M5 12h14M13 6l6 6-6 6"/>',
    'file': '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v5h6M16 13H8M16 17H8"/>',
    'edit': '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    'source': '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
    'close': '<path d="M18 6 6 18M6 6l12 12"/>',
}


def icon(name):
    return f'<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">{ICONS[name]}</svg>'


# Constructor parameters shown in reference tables. Every public parameter needs a
# description, so a new SDK argument cannot publish as an unexplained signature.
_CONNECTION = {
    'base_url': 'Welt API address. None uses WELT_BASE_URL, then https://welt.ergodic.dev.',
    'api_key': 'Explicit key. None uses WELT_API_KEY, then access saved by welt login or connect(save=True).',
    'workspace': 'Workspace ID or slug that the credential must belong to. None uses WELT_WORKSPACE when set.',
}
_CLIENT = {**_CONNECTION,
           'timeout': 'HTTP timeout in seconds for each request.',
           'transport': 'Optional httpx transport, for tests or custom networking.',
           'max_retries': 'Bounded retries for GET requests, 0 to 5. Mutations never retry automatically.',
           'max_retry_delay': 'Longest wait between GET retries, 0 to 60 seconds.'}
_ESTIMATOR = {'model': 'Catalogue model ID. Changing it needs a new fit; see Models and limits.',
              'configuration': 'Named model configuration from the catalogue.',
              'base_url': _CONNECTION['base_url'], 'api_key': _CONNECTION['api_key'],
              'random_state': 'Seed sent with the fit and pinned with the predictor.',
              'timeout': 'Seconds a blocking call waits locally. When it runs out, the server job keeps running.'}
PARAMS = {
    'Client': _CLIENT, 'AsyncClient': _CLIENT,
    'Job': {'client': 'The Client that owns the HTTP connection.', 'job_id': 'Durable job ID returned at submission.'},
    'AsyncJob': {'client': 'The AsyncClient that owns the HTTP connection.', 'job_id': 'Durable job ID returned at submission.'},
    'Classifier': _ESTIMATOR, 'Regressor': _ESTIMATOR,
    'CausalDiscovery': {**_ESTIMATOR, 'model': 'Catalogue discovery model ID: arrow, cdfm or avici.',
                        'random_state': 'Unsigned 32-bit seed recorded with the discovery job.'},
    'CausalResult': {'payload': 'Native result payload. Usually obtained from discover() or causal_result(), not built by hand.'},
    'Forecast': {'model': 'Forecast model ID.', 'configuration': "Model configuration. This version accepts only 'default'.",
                 'base_url': _CONNECTION['base_url'], 'api_key': _CONNECTION['api_key'], 'workspace': _CONNECTION['workspace'],
                 'timeout': 'Seconds a blocking prediction waits locally. The run keeps going when it runs out.'},
    'ForecastResult': {'payload': 'Native forecast payload, validated on construction.', 'job_id': 'ID of the run that produced it.'},
    'ForecastJob': {'job': 'Underlying Job handle.', 'expected': 'Request expectations the result must match.',
                    'owns_client': 'True when closing this job should also close its client.'},
    'AsyncForecastJob': {'job': 'Underlying AsyncJob handle.'},
}

# Every documented method gets a return contract and an actionable error boundary.
METHODS = {
 'connect': ('Explicitly approve workspace access through browser/device authorization; no constructor login.', 'The same connected Client. Python defaults to process-only origin-bound access; save=True opts into private local persistence.', 'AuthenticationError for denied/expired/consumed/unsafe responses or storage, PermissionDeniedError for an existing insufficient key, TransportError or local KeyboardInterrupt; printed link supports headless/remote notebooks.'),
 'request': ('Advanced raw versioned HTTP resource access; prefer named public methods for validation and contracts.', 'Decoded JSON value, or None for a 204 endpoint (await for AsyncClient).', 'Sanitized typed HTTP/transport errors; authenticated routes require credentials. GET retries are bounded; POST mutations never auto-retry.'),
 'fit': ('Prepare a new immutable predictor; wait for durable remote preparation.', 'The same fitted estimator (self).', 'ValueError for schema/target conflicts before upload; authentication, input, unavailable-model, capacity, execution or JobTimeoutError during remote preparation.'),
 'submit_fit': ('Submit durable preparation immediately. Estimator submission uploads; Client submission reuses a dataset ID.', 'Job (or awaited AsyncJob); estimator jobs own job.client, which callers close explicitly.', 'Schema/target validation, authentication, permissions, limits, availability or capacity errors. Retain the returned job ID before waiting.'),
 'predict': ('Predict on feature-only query rows with the pinned task and fitted version.', 'Estimator: numpy.ndarray, shape (n_rows,). Client: provenance-bearing result dictionary.', 'Unfitted estimator, schema mismatch, authentication, unavailable worker, capacity or execution errors; pending prediction preserves a job ID.'),
 'predict_proba': ('Return native class probabilities in classes_ column order.', 'numpy.ndarray, shape (n_rows, n_classes). No calibrated coverage guarantee.', 'Prediction errors; ValueError if the model returns no probabilities.'),
 'predict_details': ('Return predictions plus operation/model/version provenance.', 'JSON-compatible result dictionary; probabilities requested explicitly.', 'Same validation and remote errors as predict.'),
 'from_predictor': ('Reopen an owned predictor using the current credentials, without a new fit or training-table upload.', 'A fitted estimator of the matching task, with pinned model/configuration/version/seed.', 'NotFoundError for missing/foreign identity; authentication/permission errors; ValueError for a task mismatch.'),
 'metadata': ('Inspect a fitted predictor under current workspace authorization.', 'Server predictor metadata dictionary.', 'Unfitted estimator, authentication, permission, or NotFoundError.'),
 'export_metadata': ('Export provenance only; no weights, executable context or credentials.', 'JSON-compatible server metadata dictionary.', 'Same errors as metadata.'),
 'models': ('Inspect service-authoritative model task profiles, versions, readiness, configurations and limits.', 'List of model dictionaries. Presence alone does not qualify execution.', 'Transport or service error; public catalogue reads do not require credential preflight.'),
 'upload': ('Create a private workspace dataset from explicit columns and rows; target=None keeps every column for discovery.', 'Dataset dictionary with a durable id.', 'Authentication, permission, schema/target, or workload-limit errors.'),
 'job': ('Reconnect to durable work by ID. This is a regular factory on both Client and AsyncClient.', 'Job or AsyncJob handle; no request until inspection/waiting.', 'Subsequent methods require current authorization; a local timeout never cancels work.'),
 'inspect': ('Read the durable job state and server-reported stages.', 'Job metadata dictionary (await for AsyncJob).', 'Authentication, permissions, NotFoundError, transport or service errors.'),
 'result': ('Poll until completion within a local waiting budget; never resubmit or implicitly cancel.', 'Supervised fit: predictor dictionary. Predict or model_run: result dictionary. Discovery: CausalResult. Batch prediction: manifest dictionary for succeeded or partially_completed jobs, with row_count, successful_ranges, failed_ranges, result_reference and expires_at. Ranges are zero-based [start, stop), with stop excluded. Inspect failed_ranges before using the result; retrieve ordered predictions, probabilities and per-range errors separately through client.batch_payload(job.id, result_reference=manifest["result_reference"]). Failed rows have null predictions; partial completion is not success for every row.', 'JobTimeoutError with job_id for reconnect; typed failed/cancelled job errors. Synchronous request phase budgets are cooperative, not a hard wall-clock interrupt. Batch payload retrieval requires current workspace authorization and unexpired retention; no third-party download URL is followed.'),
 'cancel': ('Explicitly request remote cancellation of this job.', 'Server job metadata dictionary (await for AsyncJob).', 'Authentication, permissions, NotFoundError or job conflict; cancellation races with completion.'),
 'research_datasets': ('Search the research catalogue without uploading or fitting data.', 'Page dictionary: items, next_cursor, total, catalogue_version, facets.', 'Invalid filters/limit, authentication, permission or transport errors.'),
 'research_dataset': ('Inspect licence, schema, recorded targets/splits and eligible content before download.', 'Full research detail dictionary; content=None means bytes are not eligible.', 'Authentication/permission or NotFoundError; never infer redistribution rights from catalogue presence.'),
 'download_research_dataset': ('Download version-pinned canonical bytes with exact size/SHA verification to a caller-chosen new path. metadata=True also writes the catalogue detail to <path>.metadata.json after the verified file.', 'pathlib.Path after atomic verified publication (default). metadata=True: frozen ResearchDownload with path, metadata_path, version, sha256 and the detail dictionary. No automatic extraction or upload.', 'FileExistsError for existing destination/symlink (with metadata=True, either file is checked before any request); invalid size/version, integrity or TransportError. Interrupted downloads leave no destination.'),
 'load_research_dataset': ('Download verified content into a private temporary directory and read it as a pandas DataFrame; version=None uses the current content version. Requires the welt-client[research] extra.', 'pandas.DataFrame, or (DataFrame, ResearchMetadata) with metadata=True. A notice ZIP also exposes license_text/attribution_text and emits ResearchNoticeWarning. No file is left behind.', 'OptionalDependencyError (an ImportError) without pandas/pyarrow; content unavailable, version mismatch or size above max_bytes before download; invalid_research_archive for unexpected ZIP members; integrity or TransportError.'),
 'datasets': ('List private workspace datasets visible to the current key.', 'List of dataset dictionaries.', 'Authentication, permission or transport errors.'),
 'predictors': ('List durable fitted predictors in the current workspace.', 'List of predictor dictionaries.', 'Authentication, permission or transport errors.'),
 'jobs': ('List durable jobs in the current workspace.', 'List of job dictionaries.', 'Authentication, permission or transport errors.'),
 'usage': ('Inspect operation, lifecycle and attempt workload metadata; count operation IDs rather than events.', 'List of usage dictionaries, including nullable credits and rate_card_version accounting fields. credits records the operation\'s reservation until settlement, then its net credit charge; null means unpriced, not a promise of free execution, and differs from zero. Usage contains no raw prediction payloads or quality scores. Read current rates with pricing() and manage credits in Workspace Billing.', 'Authentication, permission or transport errors.'),
 'dataset': ('Inspect one owned private dataset.', 'Dataset metadata dictionary.', 'Authentication, permissions or NotFoundError for missing/foreign identity.'),
 'predictor': ('Inspect one owned pinned predictor.', 'Predictor metadata dictionary.', 'Authentication, permissions or NotFoundError for missing/foreign identity.'),
 'delete_dataset': ('Explicitly delete an owned dataset only when no retained predictor/result or active-job dependencies remain.', 'None; repeat owned deletion is safe.', 'NotFoundError for missing/foreign identity; ConflictError code dependency_conflict for dependencies. Never automatic cleanup.'),
 'delete_causal_result': ('Explicitly delete an owned primary native graph and scores; job/provenance/usage remain.', 'None; repeat owned deletion is safe.', 'NotFoundError for missing/foreign identity; retryable deletion_pending keeps quota/dependencies reserved. Deleted reads raise ResultDeletedError.'),
 'submit_discover': ('Submit target-free observational discovery without converting the native graph.', 'Job/AsyncJob. Facade jobs own job.client until explicitly closed.', 'Schema/unsupported constraints, task/model/worker availability, authentication, capacity or execution errors.'),
 'discover': ('Submit and wait for native observational discovery. Facade closes its temporary client.', 'CausalResult with native graph type, marks, variables, scores reference and pinned provenance.', 'Discovery validation/remote errors and JobTimeoutError. Timeout preserves durable work.'),
 'causal_result': ('Reopen an owned discovery job result and validate identity/provenance.', 'Immutable CausalResult with job_id.', 'NotFoundError, ResultDeletedError, authentication or InvalidCausalResultError; no graph repair.'),
 'causal_scores': ('Fetch separate native scores while preserving original axes and score semantics.', 'Score dictionary with values, variables, source_row_target_column axes and score_reference.', 'Authorization, deleted/missing result or InvalidCausalResultError; no thresholding or graph rewriting.'),
 'to_dict': ('Make an independent ordinary copy of the original native wire payload.', 'Fresh dictionary retaining graph type, endpoint marks and provenance.', 'No remote request and no hidden client or credentials.'),
 'to_ergodic': ('Convert only a declared native DAG locally, preserving isolates and direction; never project directed graphs.', 'Approved Ergodic MixedGraph. Use to_adjacency(order=list(result.variables)) for original axes.', 'OptionalDependencyError for missing/unrelated co-release dependency; UnsupportedGraphConversionError for every non-declared-DAG graph.'),
 'get_params': ('Inspect sklearn constructor parameters; credentials remain redacted.', 'Parameter dictionary; includes nested parameters when deep=True.', 'No remote request. Do not save runtime credentials; pickle drops explicit keys.'),
 'set_params': ('Set sklearn constructor parameters; a changed model/configuration requires a new fit.', 'The same estimator (self).', 'ValueError for an invalid parameter. Does not mutate existing pinned server artifacts.'),
 'score': ('Use the inherited sklearn scoring convention on explicit held-out rows.', 'Classifier: accuracy. Regressor: R². This is not an automatic hosted benchmark.', 'Prediction/schema/task errors; caller owns the split and evaluation protocol.'),
 'close': ('Close the synchronous HTTP client; retained remote resources remain.', 'None.', 'Closing is local, not deletion or cancellation.'),
 'aclose': ('Close the asynchronous HTTP client.', 'Awaited None.', 'Closing is local, not deletion or cancellation.'),
}

METHODS.update({'workspace': ('Inspect the credential workspace, role, metadata and resolved limits.', 'Workspace dictionary (await for AsyncClient).', 'Missing/invalid credential or workspace selection mismatch; no account-wide authority.'), 'pricing': ('Read public pricing without purchasing.', 'Public pricing dictionary (await for AsyncClient).', 'Transport/server availability; no payment authority.'), 'create_upload': ('Declare immutable file format/size/checksum/schema/target before sending chunks.', 'Upload session dictionary with ID and chunk_bytes; save its ID for recovery.', 'Server envelope/capacity/parser availability; no automatic mutation replay.'), 'upload_session': ('Inspect an owned upload and acknowledged chunks.', 'Upload metadata/progress dictionary.', 'NotFoundError for another tenant; explicit expired/failed status.'), 'put_upload_chunk': ('Send bytes with SHA256 for an explicit chunk index.', 'Updated upload progress dictionary.', 'ValueError for invalid index/bytes; conflicts for changed content; transport errors do not silently retry.'), 'complete_upload': ('Validate owned chunks/checksum/schema and create immutable dataset.', 'Dataset dictionary.', 'Incomplete/invalid/expired uploads and workload/parser limits; inspect before retrying.'), 'upload_file': ('Hash a local CSV/Parquet file, verify resume metadata, skip acknowledged chunks and complete.', 'Dataset dictionary; uses bounded chunk memory.', 'Local file/metadata ValueError; remote errors. Use create_upload and retain its ID for recovery. Async local hashing/reads are blocking disk I/O.'), 'submit_batch_predict': ('Submit durable prediction on an owned dataset using a pinned predictor.', 'Job/AsyncJob with durable ID.', 'Availability/envelope/capacity and idempotency conflicts; does not silently resubmit.'), 'job_result': ('Read ready job result without waiting.', 'Fit/prediction/native graph/batch result dictionary.', 'Not-ready, expired/deleted/not-owned errors retain typed code.'), 'batch_payload': ('Retrieve retained batch JSON through the same authenticated origin.', 'Payload dictionary with ordered predictions/errors/provenance.', 'Not-ready/not-owned/expired result or wrong reference; no third-party URL fetching.')})

METHODS.update({
 'model': ('Inspect a catalogue model or explicitly named hidden model under current workspace authority.', 'Model metadata dictionary; hidden entries carry tasks, limits, resolved version/configuration, rights and availability.', 'NotFoundError for unknown or licence-restricted entries; unavailable native worker remains explicit.'),
 'submit_model_run': ('Submit one bounded hidden-model run with an explicit task payload and replay key.', 'Job or awaited AsyncJob. Job.result() returns the task-specific result dictionary.', 'Input/envelope, licence scope, worker availability, capacity or idempotency conflicts. No fitted predictor is created; resubmission with a new key executes again.'),
 'delete_model_run': ('Explicitly purge an owned terminal model run input and result.', 'Deletion dictionary {id, deleted: true}; job identity, idempotency and usage remain.', 'NotFoundError for foreign/missing/non-model-run jobs; conflict for an active run. Subsequent result retrieval is NotFoundError.')})


METHODS.update({
 'forecast_job': ('Reconnect to a typed forecast run without submitting work; a regular factory on both clients.', 'ForecastJob or AsyncForecastJob borrowing this client; typed ForecastResult after waiting.', 'Task/identity/provenance validation on inspect/result/cancel; authorization and remote errors. Reconnect cannot reconstruct the original request context.'),
 'predict_quantiles': ('Submit one guarded zero-shot forecast run with owned history and native quantile levels.', 'numpy.ndarray, shape (horizon, n_quantiles), in requested level order.', 'Unfitted or changed context/connection, invalid horizon/levels, version conflicts, availability, capacity, execution, JobTimeoutError or InvalidForecastResultError.'),
 'to_frame': ('Copy native future UTC rows into a DataFrame; no remote call.', 'pandas.DataFrame with median and q_<level> columns.', 'No result repair, calibrated coverage or evaluation claim.'),
 'to_dict': ('Return an independent JSON-compatible copy of the native result.', 'Dictionary preserving native result fields; job_id remains a separate property.', 'No remote call; mutations do not affect the immutable result.')})
FORECAST_METHODS = {
 'fit': ('Copy and validate local univariate history, then resolve workspace/model versions. No history upload, predictor or training job.', 'The same fitted Forecast (self); zero-shot context only.', 'ValueError before transport for invalid history/time/parameters; authentication, metadata or unsupported-capability errors. Failed refit clears old context.'),
 'predict': ('Submit one guarded run with copied history; wait within timeout.', 'numpy.ndarray, shape (horizon,), containing native medians.', METHODS['predict_quantiles'][2]),
 'predict_details': ('Submit one guarded run, wait and validate native provenance and exact request expectations.', 'Immutable ForecastResult; properties and conversions share this one run.', METHODS['predict_quantiles'][2]),
 'submit_predict': ('Submit one guarded durable forecast without waiting; retain job.id.', 'ForecastJob owning its new client; close explicitly or use a context manager.', METHODS['predict_quantiles'][2])}
TYPED_FORECAST_METHODS = {
 'result': ('Wait within a local budget, validate durable task/version/result identities and preserve native values.', 'ForecastResult (await for AsyncForecastJob). Estimator jobs also validate exact submitted context/horizon/future times/levels; reconnects cannot reconstruct missing original context.', 'Typed remote errors, JobTimeoutError retaining job_id, or InvalidForecastResultError. No resubmission, repair or implicit cancellation.'),
 'cancel': ('Validate the forecast task/identity before explicitly requesting remote cancellation.', 'Validated server job metadata (await for AsyncForecastJob).', 'Task/version/provenance validation, authentication, permission, NotFoundError or job conflict; cancellation may race completion.'),
 'close': ('Close an estimator-submitted job’s owned client. Reconnected handles borrow and leave their client open.', 'None; no cancellation or deletion.', 'Local cleanup only.'),
 'aclose': ('Finish a reconnected async handle; its borrowed AsyncClient remains open.', 'Awaited None; no cancellation or deletion.', 'Local cleanup only.')}


def public_signature(obj):
    signature = inspect.signature(obj)
    return signature.replace(parameters=[parameter for name, parameter in signature.parameters.items()
                                         if not name.startswith("_") and name not in ("self", "cls")])


def format_signature(label, signature):
    """One line when short; otherwise one parameter per line so nothing scrolls sideways."""
    text = f'{label}{signature}'
    if len(text) <= 72 or not signature.parameters:
        return text
    items, keyword_only = [], False
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_POSITIONAL:
            keyword_only = True
        elif parameter.kind is parameter.KEYWORD_ONLY and not keyword_only:
            items.append('*')
            keyword_only = True
        items.append(str(parameter))
    tail = str(signature)[str(signature).rindex(')') + 1:]
    return f'{label}(\n' + ''.join(f'    {item},\n' for item in items) + ')' + tail


def parameter_table(name, signature):
    described = PARAMS.get(name, {})
    actual = [parameter for key, parameter in signature.parameters.items() if not key.startswith('_')]
    missing = [parameter.name for parameter in actual if parameter.name not in described]
    stale = set(described) - {parameter.name for parameter in actual}
    if missing or stale:
        raise SystemExit(f'Reference parameters for {name} need descriptions: missing {missing}, stale {sorted(stale)}')
    if not actual:
        return []
    rows = ['| Parameter | Default | Description |', '| --- | --- | --- |']
    for parameter in actual:
        default = 'required' if parameter.default is parameter.empty else f'`{parameter.default!r}`'
        rows.append(f'| `{parameter.name}` | {default} | {described[parameter.name]} |')
    return ['\n'.join(rows)]


def reference():
    parts = ['# Python reference', 'Signatures below are inspected from the installed matching SDK wheel. Methods are grouped by their task. [Guides](own-data.md) teach complete workflows; this page describes returns and failures. Async resource methods use the same contracts with `await`, except the regular `job(id)` factory.', '## Authentication and errors', 'Authenticated operations use an explicit constructor key or `WELT_API_KEY`; missing-key guidance is local. Missing, invalid/expired, and insufficient-permission failures remain distinct. Never log credentials or raw row payloads. Errors expose safe `code`, `request_id`, `job_id`, `status_code`, `retryable` and `retry_after` metadata when available. See [waiting and recovery](async-and-errors.md) for typed actions.']
    for name in REQUIRED:
        obj = getattr(welt, name)
        parts += [f'## {name}', f'```python\n{format_signature(name, inspect.signature(obj))}\n```', *parameter_table(name, inspect.signature(obj))]
        if name == 'CausalResult':
            parts.append('Immutable native result. Read `id`, `dataset_id`, `job_id`, `variables`, `edge_marks`, `native_graph_type`, `model_version`, `configuration_version`, `score_reference`, `decoder`, `assumptions` and `diagnostics`. The result retains no HTTP client or credentials. Scores remain a separate authenticated read.')
        if name == 'Forecast':
            parts.append('Zero-shot local history context, not a durable Predictor. See [forecasting](forecasting.md) for input/time/version/ownership rules; no score or automatic cross-validation contract.')
        if name == 'ForecastResult':
            parts.append('Immutable native result with job_id, model_id, model_version, configuration, timestamps, predictions, quantile_levels and quantiles properties. Arrays/dictionaries are owned copies; native medians and quantiles carry no calibrated coverage claim.')
        for key, method in inspect.getmembers(obj, inspect.isroutine):
            supported_inherited_score = name in ('Classifier', 'Regressor') and key == 'score'
            if key.startswith('_') or not (method.__module__.startswith('welt') or supported_inherited_score):
                continue
            overrides = FORECAST_METHODS if name == 'Forecast' else TYPED_FORECAST_METHODS if name in ('ForecastJob', 'AsyncForecastJob') else {}
            if key not in METHODS and key not in overrides:
                raise SystemExit(f'Missing annotated method contract: {name}.{key}')
            purpose, returns, errors = (html.escape(text, quote=False) for text in (overrides[key] if key in overrides else METHODS[key]))
            parts += [f'### {name}.{key}', purpose, f'```python\n{format_signature(name + "." + key, public_signature(method))}\n```', f'**Returns:** {returns}\n{{: .contract }}', f'**Errors and limits:** {errors}\n{{: .contract }}']
    parts += ['## Error types', 'All remote errors inherit `WeltError`. Table/schema validation may raise ordinary `ValueError` before upload. Do not catch every error and silently fit again.', '| Type | Next action |', '| --- | --- |']
    types = {'AuthenticationError': 'Configure a missing key locally; replace an invalid, expired or revoked key in Account.', 'PermissionDeniedError': 'Use a key with the required workspace write/fit/predict permission.', 'InvalidInputError': 'Check the named schema, explicit target and task limits.', 'ModelUnavailableError': 'Inspect current task_profiles and worker readiness; no automatic model fallback.', 'CapacityError / RateLimitError': 'Respect retry_after and workspace admission limits; inspect existing work.', 'JobTimeoutError / PredictionPendingError': 'Retain job_id and reconnect; local timeout does not cancel.', 'ExecutionError / JobCancelledError': 'Inspect the durable job and safe request/job IDs; intentional retry needs a new decision.', 'NotFoundError / ConflictError': 'Check workspace ownership and resource dependencies.', 'TransportError': 'Inspect existing job state before replaying a mutation; GET retries are bounded.', 'ResultExpiredError / ResultDeletedError': 'Temporary access expiry and explicit primary deletion are distinct.', 'InvalidCausalResultError': 'Reject malformed native payload without repair.', 'UnsupportedGraphConversionError / OptionalDependencyError': 'Preserve native graph; convert declared DAGs only with approved local co-release dependency. For load_research_dataset, install welt-client[research].'}
    types['InvalidForecastResultError'] = 'Reject malformed, crossed or mismatched native forecast payload without repair.'
    types['WeltError (HTTP 402)'] = 'Inspect status_code and code: insufficient_credits, spend_cap_reached or plan_past_due. Open Workspace Billing to review available credits and pending reservations, the spend cap or unpaid plan. Resolve the cause before intentionally retrying; this is not an automatic backoff or SDK payment operation.'
    parts += [f'| `{name}` | {meaning} |' for name, meaning in types.items()]
    return '\n\n'.join(parts[:-len(types)-2]) + '\n\n' + '\n'.join(parts[-len(types)-2:])


FENCE = re.compile(r'^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^\n`~]*)\n(?P<code>.*?)\n?^(?P=indent)(?P=fence)[ \t]*$', re.M | re.S)
PYTHON_TOKEN = re.compile(r'("[^"\n]*"|\x27[^\x27\n]*\x27|#[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b)')
PYTHON_KEYWORDS = frozenset('from import with as if elif else for while in not and or is None True False await async try except '
                            'finally raise assert return def class lambda yield pass break continue'.split())
CALLOUTS = {'note': 'Note', 'tip': 'Tip', 'caution': 'Caution', 'warning': 'Caution', 'need': "You'll need"}


def highlight(raw, language):
    """Highlight literal code without executing it or adding a runtime dependency."""
    if language not in ('python', 'py', 'sh', 'bash', 'shell'):
        return html.escape(raw)
    if language in ('sh', 'bash', 'shell'):
        lines = []
        for line in raw.splitlines(keepends=True):
            content = line.rstrip('\n')
            ending = line[len(content):]
            lines.append((f'<span class="co">{html.escape(content)}</span>' if content.lstrip().startswith('#') else html.escape(content)) + ending)
        return ''.join(lines)
    pieces, cursor = [], 0
    for item in PYTHON_TOKEN.finditer(raw):
        value = item[0]
        kind = ('co' if value.startswith('#') else 'st' if value.startswith(('"', "'")) else 'nu' if value[0].isdigit()
                else 'kw' if value in PYTHON_KEYWORDS else 'fn' if raw[item.end():item.end() + 1] == '(' else '')
        if not kind:
            continue
        pieces += [html.escape(raw[cursor:item.start()]), f'<span class="{kind}">{html.escape(value)}</span>']
        cursor = item.end()
    pieces.append(html.escape(raw[cursor:]))
    return ''.join(pieces)


def mark_lines(highlighted, spec, count):
    """Wrap 1-based highlighted line ranges such as "2-3,7"; other lines stay unchanged."""
    if not spec:
        return highlighted
    selected = set()
    for part in spec.split(','):
        start, _, stop = part.strip().partition('-')
        if not start.isdigit() or (stop and not stop.isdigit()) or not 1 <= int(start) <= int(stop or start) <= count:
            raise SystemExit(f'Invalid highlighted line range: {spec!r}')
        selected.update(range(int(start), int(stop or start) + 1))
    return '\n'.join(f'<span class="hl">{line or " "}</span>' if number in selected else line
                     for number, line in enumerate(highlighted.split('\n'), 1))


def fence_info(info):
    language, _, rest = info.strip().partition(' ')
    if not re.fullmatch(r'(?:\s*[a-z]+="[^"]*")*\s*', rest):
        raise SystemExit(f'Malformed code fence attributes in: {info!r}')
    attributes = dict(re.findall(r'([a-z]+)="([^"]*)"', rest))
    if set(attributes) - {'file', 'tab', 'hl', 'title'}:
        raise SystemExit(f'Unsupported code fence attribute in: {info!r}')
    return (language or 'text'), attributes


def code_block(blocks, position=0):
    """Render one fence, or adjacent tab="…" fences as one accessible tab group."""
    if len(blocks) == 1 and blocks[0][0] == 'output':
        _, attributes, raw = blocks[0]
        label = html.escape(attributes.get('title', 'Output'))
        return (f'<div class="output"><div class="output-head">{label}</div>'
                f'<pre tabindex="0" aria-label="{label}"><code>{html.escape(raw)}</code></pre></div>')
    panes = [(language, attributes, f'<pre tabindex="0" aria-label="{html.escape(attributes.get("tab") or language, quote=True)} code">'
              f'<code>{mark_lines(highlight(raw, language), attributes.get("hl"), len(raw.splitlines()))}</code></pre>')
             for language, attributes, raw in blocks]
    if len(panes) == 1:
        language, attributes, pane = panes[0]
        label = attributes.get('file') or language
        icon = ' code-file' if 'file' in attributes else ''
        return f'<div class="code-wrap"><div class="code-head"><span class="code-language{icon}">{html.escape(label)}</span></div>{pane}</div>'
    key = 'tabs-' + hashlib.sha256(json.dumps([position, blocks]).encode()).hexdigest()[:10]
    buttons, panels = [], []
    for index, (_, attributes, pane) in enumerate(panes):
        selected = index == 0
        buttons.append(f'<button id="{key}-{index}-tab" type="button" role="tab" aria-selected="{str(selected).lower()}" '
                       f'aria-controls="{key}-{index}-panel" tabindex="{0 if selected else -1}">{html.escape(attributes["tab"])}</button>')
        panels.append(f'<div id="{key}-{index}-panel" role="tabpanel" aria-labelledby="{key}-{index}-tab"'
                      + ('' if selected else ' hidden') + f'>{pane}</div>')
    return ('<div class="code-wrap code-tabbed"><div class="code-head"><div class="code-tabs" role="tablist" aria-label="Code variants">'
            + ''.join(buttons) + '</div></div>' + ''.join(panels) + '</div>')


def extract_code(text, nonce):
    """Replace fences with placeholders; adjacent tab fences at one indent form one group."""
    rendered, output, cursor = [], [], 0
    matches = list(FENCE.finditer(text))
    index = 0
    while index < len(matches):
        match = matches[index]
        indent = match['indent']
        if len(indent.expandtabs(4)) % 4:
            raise SystemExit('Indent fenced code by a multiple of four spaces (inside lists and callouts).')
        group = [match]
        if 'tab="' in match['info']:
            while (index + 1 < len(matches) and 'tab="' in matches[index + 1]['info'] and matches[index + 1]['indent'] == indent
                   and not text[group[-1].end():matches[index + 1].start()].strip()):
                index += 1
                group.append(matches[index])
        blocks = []
        for item in group:
            language, attributes = fence_info(item['info'])
            raw = '\n'.join(line[len(indent):] if line.startswith(indent) else line for line in item['code'].split('\n'))
            blocks.append((language, attributes, raw + '\n' if raw else raw))  # Fenced code keeps its final newline.
        if len(blocks) > 1 and any(language == 'output' for language, _, _ in blocks):
            raise SystemExit('Output panels cannot be tabbed.')
        token = f'WELTCODE{nonce}N{len(rendered)}X'
        rendered.append(code_block(blocks, len(rendered)))
        output += [text[cursor:group[0].start()], f'\n{indent}{token}\n']
        cursor = group[-1].end()
        index += 1
    output.append(text[cursor:])
    return ''.join(output), rendered


def wrap_steps(body):
    """Group consecutive `## Title {.step}` sections into a numbered steps list."""
    starts, depth, cursor = [], 0, 0
    for match in re.finditer(r'<(/?)(div|details|section|blockquote|ul|ol|li|table)\b[^>]*>|<h2 [^>]*>', body):
        if match[0].startswith('<h2 '):
            if depth == 0:
                starts.append(match.start())
            elif re.search(r'class="[^"]*\bstep\b', match[0]):
                raise SystemExit('Step headings must be top-level, not inside a callout, list or details block.')
        else:
            depth += -1 if match[1] else 1
    parts = [body[a:b] for a, b in zip([0, *starts], [*starts, len(body)]) if body[a:b]]
    result, open_steps = [], False
    for part in parts:
        is_step = re.match(r'<h2 [^>]*class="[^"]*\bstep\b', part) is not None
        if is_step and not open_steps:
            result.append('<div class="steps">')
            open_steps = True
        if not is_step and open_steps and part.startswith('<h2 '):
            result.append('</div>')
            open_steps = False
        if is_step:  # The section carries the step; the heading keeps any other classes.
            part = re.sub(r'^(<h2 [^>]*?) class="([^"]*)"', lambda m: m[1] + (f' class="{c}"' if (c := ' '.join(x for x in m[2].split() if x != 'step')) else ''), part, count=1)
        result.append(f'<section class="step">{part}</section>' if is_step else part)
    if open_steps:
        result.append('</div>')
    return ''.join(result)


def render_markdown(text):
    text = re.sub(r'\]\(([^():#]+)\.md(#[^)]*)?\)', lambda m: '](' + m[1] + '.html' + (m[2] or '') + ')', text)
    nonce = secrets.token_hex(8)  # Placeholders cannot collide with page text or code.
    text, blocks = extract_code(text, nonce)
    renderer = markdown.Markdown(extensions=['tables', 'toc', 'admonition', 'attr_list', 'md_in_html'],
                                 extension_configs={'toc': {'permalink': False}})
    body = renderer.convert(text)
    token = f'WELTCODE{nonce}N(\\d+)X'
    # One pass, so inserted code is never rescanned; an indented placeholder may arrive as code.
    body = re.sub(rf'<pre><code>{token}\s*</code></pre>|<p>{token}</p>|{token}',
                  lambda m: blocks[int(next(group for group in m.groups() if group is not None))], body)
    body = body.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable table"><table>').replace('</table>', '</table></div>')

    def heading(match):
        level, attributes, content = match[1], match[2], match[3]
        identifier = re.search(r' id="([^"]+)"', attributes)
        if not identifier:
            return match[0]
        rest = re.sub(r' id="[^"]+"', '', attributes)
        label = html.escape(re.sub('<[^>]+>', '', content), quote=True)
        return (f'<h{level} id="{identifier[1]}"{rest}>{content}<a class="anchor" href="#{identifier[1]}" '
                f'aria-label="Link to {label}">#</a></h{level}>')
    body = re.sub(r'<h([23])((?: [\w-]+="[^"]*")*)>(.*?)</h\1>', heading, body, flags=re.S)

    def callout(match):
        kinds = match[1].split()
        if len(kinds) != 1 or kinds[0] not in CALLOUTS:
            raise SystemExit(f'Unsupported callout type: {match[1]}')
        kind = kinds[0]
        title = match[2]
        if title is not None and title == kind.capitalize():  # No explicit title: use the component's label.
            title = html.escape(CALLOUTS[kind])
        heading = '' if title is None else f'<p class="callout-title">{title}</p>'
        return f'<div class="callout {"caution" if kind == "warning" else kind}" role="note">\n{heading}'
    body = re.sub(r'<div class="admonition ([^"]*)">\s*(?:<p class="admonition-title">(.*?)</p>)?', callout, body, flags=re.S)
    # A short opening paragraph reads as the page lead; long ones stay body text.
    body = re.sub(r'(</h1>\s*)<p>((?:(?!</p>).){1,320}?</p>)', r'\1<p class="lead">\2', body, count=1, flags=re.S)
    return wrap_steps(body), renderer.toc_tokens


def search_records(body, name, title):
    """Index rendered heading IDs, including each reference method's exact anchor."""
    def plain(value):
        value = html.unescape(re.sub(r'<[^>]+>', '', value))
        value = re.sub(r'\{\{actions:[^}]+\}\}', '', value)
        return ' '.join(value.split())

    heading_pattern = r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>'
    headings = [item for item in re.finditer(heading_pattern, body, re.S)
                if item[1] == '2' or name == 'reference']
    intro_end = headings[0].start() if headings else len(body)
    records = [dict(title=title, text=plain(body[:intro_end]), href=name + '.html')]
    for index, item in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        heading = plain(item[3].split('<a class="anchor"', 1)[0])
        record = dict(title=f'{title} · {heading}', text=plain(body[item.end():end]),
                      href=name + '.html#' + item[2])
        if name == 'reference' and item[1] == '3':
            record['method'] = heading
        records.append(record)
    return records


def current_redirect(page):
    """Explicitly redirect an old learning route while retaining query and fragment."""
    destination = '../' + page
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Open current Welt documentation</title>'
            '<link rel="canonical" href="' + PUBLIC + '/' + html.escape(page, quote=True) + '">'
            '</head><body><p>This guide now opens in the current documentation.</p>'
            '<p><a id="current-docs" href="' + html.escape(destination, quote=True) + '">Open current documentation</a></p>'
            '<script>const destination = new URL(' + json.dumps(destination) + ', window.location.href);'
            'destination.search = window.location.search;destination.hash = window.location.hash;'
            'document.getElementById("current-docs").href = destination.href;'
            'window.location.replace(destination.href);</script></body></html>')


# Published historical documentation folders (v<version>/), oldest first.
HISTORICAL_VERSIONS = ('0.6.0', '0.7.0', '0.7.1', '0.8.0', '0.9.1', '0.10.0', '0.10.1')


def release_tuple(version):
    """Numeric (major, minor, patch) of a release, tolerating pre-release/local suffixes such as 0.9.0rc1."""
    match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', version)
    if not match:
        raise ValueError(f'Unrecognised SDK version: {version!r}')
    return tuple(int(part or 0) for part in match.groups())


def browser_connect_release(version):
    """Releases from 0.8.0 onward document Client.connect() and the notebook landing."""
    return release_tuple(version) >= (0, 8, 0)


def older_versions(version):
    """Historical documentation versions older than this release, newest first."""
    return [old for old in reversed(HISTORICAL_VERSIONS) if release_tuple(old) < release_tuple(version)]


def notebook_view(source, version):
    """Render cleared notebook sources without executing cells or loading outputs."""
    notebook = json.loads(source.read_text())
    parts = ['<p><a href="notebooks.html">← All notebooks</a></p>',
             notebook_actions(source.stem, version),
             '<p class="notebook-preview-note">Static notebook preview · outputs are cleared. '
             'Open in Colab or download the notebook to run the cells. '
             '<a href="notebooks.html#run-in-colab">Colab setup</a></p>']
    title = None
    code_count = 0
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if title is None:
                heading = re.search(r'^# (.+)$', text, re.M)
                if heading:
                    title = heading[1]
            parts.append(render_markdown(text)[0])
        elif cell['cell_type'] == 'code':
            if cell.get('outputs') or cell.get('execution_count') is not None:
                raise SystemExit(f'Notebook views require cleared outputs: {source.name}')
            code_count += 1
            fence = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', text)), default=0))
            code = render_markdown(f'{fence}python\n{text.rstrip(chr(10))}\n{fence}')[0]
            code = code.replace('<div class="code-wrap">',
                                '<div class="code-wrap" data-copy="' + html.escape(text, quote=True) + '">', 1)
            parts.append(f'<section class="notebook-cell" aria-label="Code cell {code_count}">'
                         '<span class="cell-prompt" aria-hidden="true">In [ ]</span>' + code + '</section>')
        else:
            raise SystemExit(f'Unsupported notebook cell type: {source.name}')
    if title is None:
        raise SystemExit(f'Notebook view needs a title: {source.name}')
    return '<div class="notebook-view">' + ''.join(parts) + '</div>', title


def notebook_actions(name, version):
    # Colab runs the exact reviewed teaching source, independently of SDK bytes.
    revision = os.environ.get('DOCS_NOTEBOOK_REF') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise SystemExit('Notebook links require an exact documentation source commit.')
    colab = f'https://colab.research.google.com/github/ergodic-ai/welt-python/blob/{revision}/examples/notebooks/{name}.ipynb'
    label = html.escape(name.replace('-', ' '), quote=True)
    return (f'<div class="notebook-actions" role="group" aria-label="{label} notebook actions">'
            f'<a href="{colab}">Open in Colab<span class="sr-only">: {label}</span> ↗</a>'
            f'<a href="downloads/{name}.ipynb" download="{name}.ipynb">Download notebook<span class="sr-only">: {label}</span></a>'
            f'<a href="notebook-{name}.html">View<span class="sr-only">: {label}</span></a></div>')


MODELS_SNAPSHOT = ROOT / 'docs/models-snapshot.json'
PUBLIC_CATALOGUE = 'https://welt.ergodic.dev/v1/models'
TASK_LABELS = {'classification': 'Classification', 'regression': 'Regression', 'causal_discovery': 'Causal discovery'}


def available_models():
    """Available, executable task profiles from the committed public catalogue snapshot (DEC-073 #7)."""
    snapshot = json.loads(MODELS_SNAPSHOT.read_text())
    if snapshot.get('source') != PUBLIC_CATALOGUE:
        raise SystemExit(f'The Models table renders only a snapshot of {PUBLIC_CATALOGUE}.')
    def span(limits, low, high):
        if high not in limits:
            return '—'
        return f'{limits[low]}–{limits[high]}' if low in limits else f'up to {limits[high]}'
    rows = []
    for model in snapshot['models']:
        for task in model['tasks']:
            if task['status'] != 'available' or not task['execution_available']:
                continue
            limits = task['limits']
            columns = span(limits, 'min_features', 'max_features') if 'max_features' in limits else span(limits, 'min_columns', 'max_columns')
            rows.append((model['name'], model['id'], TASK_LABELS.get(task['task'], task['task']), span(limits, 'min_rows', 'max_rows'),
                         columns, span(limits, '', 'max_prediction_rows'), span(limits, '', 'max_classes')))
    if not rows:
        raise SystemExit('Models snapshot has no available task profiles.')
    return rows, snapshot['taken_at'][:10]


MODEL_COLUMNS = ('Model', 'Task', 'Training rows', 'Features or variables', 'Query rows', 'Classes')
MODEL_NOTE = ('Supervised rows count feature columns without the target; causal discovery counts every variable. '
              'The live catalogue is authoritative: Client().models() reports current limits and readiness.')


def models_table():
    rows, taken = available_models()
    body = ''.join('<tr>' + f'<td><strong>{html.escape(name)}</strong> <code>{html.escape(model_id)}</code></td>'
                   + ''.join(f'<td class="num">{html.escape(value)}</td>' if index else f'<td>{html.escape(value)}</td>' for index, value in enumerate(rest))
                   + '</tr>' for name, model_id, *rest in rows)
    return ('<div class="table-wrap models-table" tabindex="0" role="region" aria-label="Available models"><table><thead><tr>'
            + ''.join(f'<th scope="col">{label}</th>' for label in MODEL_COLUMNS) + '</tr></thead><tbody>' + body + '</tbody></table></div>'
            f'<p class="table-note">Public catalogue snapshot from {html.escape(taken)}. Supervised rows count feature columns without the target; '
            'causal discovery counts every variable. The live catalogue is authoritative: <code>Client().models()</code> reports current limits and readiness.</p>')


def models_markdown():
    rows, taken = available_models()
    lines = ['| ' + ' | '.join(MODEL_COLUMNS) + ' |', '|' + ' --- |' * len(MODEL_COLUMNS)]
    lines += [f'| {name} (`{model_id}`) | ' + ' | '.join(rest) + ' |' for name, model_id, *rest in rows]
    return '\n'.join(lines) + f'\n\nPublic catalogue snapshot from {taken}. {MODEL_NOTE}'


def guide_groups(version):
    return [(label, [(text, page) for text, page in pages if page != 'connect' or browser_connect_release(version)])
            for label, pages in GUIDES]


def pager(name, version, link):
    order = [(text, page) for _, pages in guide_groups(version) for text, page in pages]
    current = 'notebooks' if name.startswith('notebook-') else name
    pages = [page for _, page in order]
    if current not in pages:
        return ''
    index = pages.index(current)
    previous = order[index - 1] if index else ('Overview', 'index')
    following = order[index + 1] if index + 1 < len(order) else None
    items = [f'<a class="pager-prev" href="{link(previous[1])}"><small>{icon("left")}Previous</small><strong>{html.escape(previous[0])}</strong></a>']
    if following:
        items.append(f'<a class="pager-next" href="{link(following[1])}"><small>Next{icon("right")}</small><strong>{html.escape(following[0])}</strong></a>')
    return '<nav class="pager" aria-label="Previous and next">' + ''.join(items) + '</nav>'


def reference_index(link):
    """Class index for the single reference page; every link is an existing anchor."""
    groups = [('Connect', ('Client', 'AsyncClient')), ('Predict', ('Classifier', 'Regressor')),
              ('Discover', ('CausalDiscovery', 'CausalResult')), ('Forecast', ('Forecast', 'ForecastResult', 'ForecastJob', 'AsyncForecastJob')),
              ('Long-running work', ('Job', 'AsyncJob')),
              ('Errors', ())]
    listed = {name for _, names in groups for name in names}
    if listed != set(REQUIRED):
        raise SystemExit('Reference class index must list every documented class.')
    parts = []
    for label, names in groups:
        items = ''.join(f'<a class="nav-item" href="#{name.lower()}"><code>{name}</code></a>' for name in names)
        if label == 'Errors':
            items = '<a class="nav-item" href="#authentication-and-errors">Authentication and errors</a><a class="nav-item" href="#error-types">Error types</a>'
        parts.append(f'<div class="nav-group"><p class="nav-group-title">{label}</p>{items}</div>')
    return '<div class="nav-group"><a class="nav-item" href="' + link('start') + '">' + icon('left') + 'All guides</a></div>' + ''.join(parts)


def shell(body, name, title, version, prefix='', outline='', home=False, markdown_source=None):
    def link(page): return prefix + ('index.html' if page == 'index' else page + '.html')
    section = SECTION.get(name, 'Get started' if name.startswith('notebook-') else 'Guides')
    nav = ''.join(f'<a href="{link(page)}"' + (' aria-current="page"' if label == section and not home else '') + f'>{label}</a>' for label, page in NAV)
    groups = guide_groups(version)
    sidebar = ''.join('<div class="nav-group">' + f'<p class="nav-group-title">{label}</p>' + ''.join(
        f'<a class="nav-item" href="{link(page)}"' + (' aria-current="page"' if page == name or (page == 'notebooks' and name.startswith('notebook-')) else '') + f'>{text}</a>'
        for text, page in pages) + '</div>' for label, pages in groups)
    side = reference_index(link) if name == 'reference' else sidebar
    versions = f'<option value="{prefix}index.html">v{version}</option>' + ''.join(
        f'<option value="{"../" if prefix == "" else ""}v{old}/">v{old}</option>' for old in older_versions(version))
    canonical = PUBLIC + ('/' if home and prefix else f'/v{version}/' + ('index.html' if name == 'index' else name + '.html'))
    page_url = f'{PUBLIC}/v{version}/' + ('index.html' if name == 'index' else name + '.html')
    issue = (f'{REPOSITORY}/issues/new?title=' + quote(f'Docs: {title}') + '&body=' + quote(f'Page: {page_url}\n\nWhat should change?\n'))
    actions = []
    if markdown_source:
        actions.append(f'<button type="button" data-markdown="{prefix}{markdown_source}">{icon("file")}<span>Copy page as Markdown</span></button>')
        if name != 'reference':
            actions.append(f'<a href="{REPOSITORY}/blob/v{version}/docs/{name}.md">{icon("source")}<span>View source at v{version}</span></a>')
    actions.append(f'<a href="{issue}">{icon("edit")}<span>Suggest an edit</span></a>')
    rail = (f'<div class="outline-inner">{outline}</div>' if outline else '') + '<div class="page-actions">' + ''.join(actions) + '</div>'
    inline_actions = '<div class="page-actions page-actions-inline">' + ''.join(actions) + '</div>'
    crumbs = '' if home or name == 'index' else f'<p class="crumbs">{html.escape(section)}{icon("chevron")}<span>{html.escape(title)}</span></p>'
    main = (f'<main id="main" tabindex="-1" class="home">{body}</main>' if home else
            f'<div class="layout"><nav class="sidebar" aria-label="Documentation">{side}</nav>'
            f'<main id="main" tabindex="-1" class="article">{crumbs}{body}{pager(name, version, link)}{inline_actions}</main>'
            f'<aside class="outline" aria-label="On this page">{rail}</aside></div>')
    theme_boot = "try{const t=localStorage.getItem('welt-docs-theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Welt Python SDK {version}: DataFrame prediction, reusable predictors, research datasets, forecasting and native causal discovery."><meta name="color-scheme" content="light dark"><title>{html.escape(title)} · Welt Python SDK {version}</title><link rel="canonical" href="{canonical}"><link rel="icon" type="image/svg+xml" href="{prefix}assets/ergodic-mark-dark.svg"><link rel="preload" href="{prefix}assets/fonts/geist-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin><link rel="stylesheet" href="{prefix}assets/tokens.css"><link rel="stylesheet" href="{prefix}assets/docs.css"><script>{theme_boot}</script><script defer src="{prefix}assets/docs.js"></script></head><body data-search="{prefix}search-index.json"><a class="skip" href="#main">Skip to content</a><header class="topbar"><button id="menu" class="icon-button menu-button" type="button" aria-haspopup="dialog" aria-label="Open navigation">{icon("menu")}</button><a class="brand" href="{link('index')}"><img src="{prefix}assets/ergodic-mark-dark.svg" alt="" width="28" height="28"><strong>Welt</strong><span class="brand-sep" aria-hidden="true"></span><span class="brand-sub">Python SDK</span></a><nav class="topnav" aria-label="Main navigation">{nav}</nav><span class="spacer"></span><button id="search" class="search-trigger" type="button" aria-haspopup="dialog">{icon("search")}<span class="search-label">Search docs</span><kbd class="shortcut" aria-hidden="true">⌘K</kbd><span class="sr-only">Search documentation</span></button><select id="version" class="version" data-version aria-label="Documentation version">{versions}</select><button id="theme" class="icon-button" type="button" aria-label="Switch theme">{icon("theme")}</button><a class="btn btn-primary btn-sm open-console" href="https://welt.ergodic.dev">Open console</a></header>{main}<footer class="footer"><span>Welt by Ergodic · Python SDK {version}</span><span class="footer-links"><a href="https://welt.ergodic.dev">Console</a><a href="{REPOSITORY}/tree/v{version}">Release source</a><a href="https://pypi.org/project/welt-client/{version}/">PyPI</a><a href="{prefix}llms.txt">llms.txt</a></span></footer><dialog id="navigation" class="nav-dialog" aria-labelledby="navigation-title"><header><strong id="navigation-title">Documentation</strong><button type="button" class="icon-button" data-close aria-label="Close navigation">{icon("close")}</button></header><div class="nav-dialog-tools"><select class="version" data-version aria-label="Documentation version">{versions}</select><a class="btn btn-primary btn-sm" href="https://welt.ergodic.dev">Open console</a></div><nav class="nav-dialog-sections" aria-label="Sections">{nav}</nav><nav aria-label="Mobile guide navigation">{side if name == 'reference' else sidebar}</nav></dialog><dialog id="search-dialog" class="palette" aria-labelledby="search-title"><h2 id="search-title" class="sr-only">Search SDK {version}</h2><div class="palette-input">{icon("search")}<label for="search-input" class="sr-only">Task, method or error</label><input id="search-input" type="search" autocomplete="off" placeholder="Search guides, methods and errors" role="combobox" aria-expanded="false" aria-controls="search-results" aria-autocomplete="list"><button type="button" data-close class="palette-close" aria-label="Close search"><kbd>Esc</kbd></button></div><ul id="search-results" role="listbox" aria-label="Search results"></ul><p id="search-empty">Search this SDK version by task, method or error.</p><div class="palette-footer" aria-hidden="true"><span><kbd>↑</kbd><kbd>↓</kbd> to move</span><span><kbd>↵</kbd> to open</span><span>Searching v{version}</span></div></dialog><div id="announcement" class="sr-only" role="status" aria-live="polite"></div></body></html>'''


def notebook_home(prefix='', sdk_version='0.8.0'):
    """API-103: visible loading and modelling cells using current Iris headers.

    Displayed Iris rows and report bind API-071's recorded seed9 holdout, not a
    fresh benchmark. The report remains a clearly labelled earlier observation, not a fresh run.
    """
    def link(page): return prefix + page + '.html'
    def code(value, lang='python'): return render_markdown(f'```{lang}\n{value}\n```')[0]
    def tabs(group, label, entries):
        buttons=[];panels=[]
        for index,(name,body) in enumerate(entries):
            key=f'{group}-{index}'
            buttons.append(f'<button id="{key}-tab" type="button" role="tab" aria-selected="{str(index == 0).lower()}" aria-controls="{key}-panel" tabindex="{0 if index == 0 else -1}">{name}</button>')
            panels.append(f'<div id="{key}-panel" role="tabpanel" aria-labelledby="{key}-tab" tabindex="0"' + (' hidden' if index else '') + f'>{body}</div>')
        return f'<div class="task-tabs" role="tablist" aria-label="{label}">' + ''.join(buttons) + '</div>' + ''.join(panels)
    install=tabs('install','Install command',[
        ('uv',code("uv pip install 'welt-client[parquet]'",'sh')),
        ('pip',code("pip install 'welt-client[parquet]'",'sh'))])
    connect=tabs('connect','Connection method',[
        ('Notebook',code('from welt import Client\n\nClient().connect()') + '<p class="card-note">Approve the code, workspace and permissions in your browser. Running remotely? Open the printed link on your own computer. Access stays in this Python process.</p>'),
        ('Terminal',code('welt login','sh') + '<p class="card-note">Approve in your browser. The CLI saves access privately for this API origin. Run Python on the same computer.</p>'),
        ('API key',code('import os\nfrom getpass import getpass\n\nos.environ["WELT_API_KEY"] = getpass("Welt API key: ")') + '<p class="card-note">Use your own read/write key from <a href="https://welt.ergodic.dev">Workspace → API keys</a>. The key stays in this Python process environment; it is never saved in code.</p>')])
    loading='''from welt import Client

data = Client().load_research_dataset(
    "asset-d839644b6de36fc5cafdda18c5daeca3"
)
target = "target"
data.head()'''
    program='''from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from welt import Classifier

train, test = train_test_split(
    data, test_size=0.2, random_state=9, stratify=data[target]
)
X_train, y_train = train.drop(columns=target), train[target]
X_test, y_test = test.drop(columns=target), test[target]

model = Classifier(model="tabicl-v2", random_state=9)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))'''
    rows=[[0,4.6,3.6,1.0,.2,0],[1,4.3,3.0,1.1,.1,0],[2,5.0,3.2,1.2,.2,0],[3,5.8,4.0,1.2,.2,0],[4,4.4,3.0,1.3,.2,0]]
    headers=['','sepal-length','sepal-width','petal-length','petal-width','target']
    table='<div class="dataframe-output" tabindex="0" role="region" aria-label="First five current Iris rows"><table><thead><tr>' + ''.join(f'<th scope="col">{name if name else "<span class=sr-only>Row</span>"}</th>' for name in headers) + '</tr></thead><tbody>'
    for row in rows:
        table+='<tr><th scope="row">'+str(row[0])+'</th>' + ''.join(f'<td>{value}</td>' for value in row[1:])+'</tr>'
    table+='</tbody></table></div>'
    report='''              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00        10
           2       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30'''
    notebook=f'<div class="notebook-cell"><span class="cell-prompt" aria-hidden="true">In [1]</span>{code(loading)}</div><div class="notebook-output"><span class="cell-prompt" aria-hidden="true">Out [1]</span>{table}</div><div class="notebook-cell"><span class="cell-prompt" aria-hidden="true">In [2]</span>{code(program)}</div><div class="notebook-output"><span class="cell-prompt" aria-hidden="true">Out [2]</span><pre class="report-output" tabindex="0" aria-label="Recorded classification report">{html.escape(report)}</pre></div>'
    hero = (f'<section class="home-hero" aria-labelledby="home-title"><p class="eyebrow"><span class="badge mono">SDK {sdk_version}</span>Python 3.11–3.13</p>'
            f'<h1 id="home-title">From a DataFrame<br><span>to a prediction.</span></h1>'
            '<p class="lead">Upload a table to your private Welt workspace, fit a qualified model and predict from Python. '
            'The same SDK reopens predictors and runs native causal discovery.</p>'
            f'<div class="actions"><a class="btn btn-primary btn-lg" href="{link("start")}">Make your first prediction{icon("right")}</a>'
            f'<a class="btn btn-lg" href="{link("reference")}">Python reference</a></div></section>')
    steps = (f'<section class="home-band" aria-labelledby="setup-title"><div class="band-head"><h2 id="setup-title">Set up in three steps</h2>'
             f'<a href="{link("start")}">Full walkthrough{icon("right")}</a></div><div class="quickstart-cards">'
             f'<section class="start-card" aria-labelledby="install-title"><header class="card-heading"><span class="card-number" aria-hidden="true">1</span><h3 id="install-title">Install</h3><a href="{link("start")}#install">Setup guide</a></header>{install}</section>'
             f'<section class="start-card" aria-labelledby="connect-title"><header class="card-heading"><span class="card-number" aria-hidden="true">2</span><h3 id="connect-title">Connect</h3><a href="{link("connect")}">Connection guide</a></header>{connect}</section>'
             f'<section class="start-card notebook-card" aria-labelledby="predict-title"><header class="card-heading"><span class="card-number" aria-hidden="true">3</span><h3 id="predict-title">Train, predict, evaluate</h3><a href="{link("start")}">Full walkthrough</a></header>'
             f'<p class="card-note prepared-data">Load Iris below: 150 rows, four features, classes 0–2. The target column is <code>target</code>. Before downloading or fitting, open <a href="https://welt.ergodic.dev">Workspace → Billing</a> and check your credits and spending limits.</p>{notebook}'
             '<p class="card-note output-provenance">Earlier recorded Iris run · TabICL v2 · seed 9 · 120 training / 30 held-out rows. This report is one observed result, not a promised score. Fit uploads the training table; prediction runs remotely.</p>'
             '<p class="card-note">For your own table, replace <code>target</code> with your target column.</p></section></div></section>')
    tasks = ''.join(f'<a class="task-card" href="{link(page)}"><strong>{html.escape(label)}</strong><span>{html.escape(text)}</span></a>'
                    for label, page, text in TASKS)
    tasks = f'<section class="home-band" aria-labelledby="tasks-title"><div class="band-head"><h2 id="tasks-title">What do you want to do?</h2></div><div class="task-cards">{tasks}</div></section>'
    models = (f'<section class="home-band" aria-labelledby="models-title"><div class="band-head"><h2 id="models-title">Available models</h2>'
              f'<a href="{link("capabilities")}">Models and limits{icon("right")}</a></div>{models_table()}</section>')
    return hero + steps + tasks + models


def home_body(start, prefix='', sdk_version='0.7.1', browser_connect=False):
    """Current landing. Pre-0.8 landings are rendered by their pinned presentation
    baseline in sdk-docs.yml, never re-derived from today's guides (API-123)."""
    if not (browser_connect and browser_connect_release(sdk_version)):
        raise SystemExit('Landing pages before SDK 0.8 are rendered from their pinned presentation baseline.')
    return notebook_home(prefix, sdk_version)


def markdown_copy(text, version):
    """The page's Markdown for agents and "Copy page as Markdown", with build placeholders resolved."""
    def actions(match):
        name = match[1]
        colab = re.search(r'href="(https://colab[^"]+)"', notebook_actions(name, version))[1]
        return f'[Open in Colab]({colab}) · [Download notebook](downloads/{name}.ipynb) · [View](notebook-{name}.html)'
    text = re.sub(r'\{\{actions:([a-z0-9-]+)\}\}', actions, text)
    text = re.sub(r'\n\{: \.contract \}', '', text)
    return text.replace('{{models-table}}', models_markdown())


def llms_index(version, titles):
    """Plain-text index of this release's Markdown pages for agents (llms.txt convention)."""
    lines = [f'# Welt Python SDK {version}', '',
             '> Python SDK for Welt: fit qualified tabular models on DataFrames, reuse predictors, forecast, '
             'load research datasets and run native causal discovery. Each link is this release\'s Markdown source.', '',
             'Product-level agent access, including the hosted Welt service and console, is at https://welt.ergodic.dev', '']
    for label, pages in GUIDES:
        lines += [f'## {label}', '']
        lines += [f'- [{titles[page]}]({PUBLIC}/v{version}/{page}.md)' for _, page in pages if page in titles]
        lines.append('')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output', default='site'); args=parser.parse_args()
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    if installed_version('welt-client') != version or getattr(welt,'__version__',None) != version:
        raise SystemExit('Installed SDK version must match the documentation source version.')
    if any(not hasattr(welt,name) for name in REQUIRED):
        raise SystemExit('Installed SDK is missing a required documented public symbol.')
    pages={p.stem:p.read_text() for p in sorted((ROOT/'docs').glob('*.md'))}
    pages['reference']=reference()
    if any(re.search(r'github(?:usercontent)?\.com/ergodic-ai/welt-python/(?:blob/|tree/)?(?:main|feat/)', text) for text in pages.values()):
        raise SystemExit('Versioned documentation cannot link to mutable SDK source.')
    base=Path(args.output); out=base/f'v{version}'; out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'docs/assets',out/'assets',dirs_exist_ok=True)
    downloads=out/'downloads'; downloads.mkdir(exist_ok=True)
    for source in (ROOT/'examples/notebooks').glob('*'):
        if source.suffix in ('.ipynb','.py','.md'):shutil.copy2(source,downloads/source.name)
    for name in ('first_prediction.py', 'first_request.py', 'forecasting.py'):
        shutil.copy2(ROOT/'examples'/name, downloads/name)
    search=[]
    titles={}
    for name,text in pages.items():
        body,toc=render_markdown(text); title=re.search(r'^# (.+)$',text,re.M)[1]
        if name == 'notebooks':
            for source in sorted((ROOT/'examples/notebooks').glob('*.ipynb')):
                body = body.replace(f'<p>{{{{actions:{source.stem}}}}}</p>', notebook_actions(source.stem, version))
            body = '<div class="notebook-index">' + body + '</div>'
        if name == 'capabilities':
            if '<p>{{models-table}}</p>' not in body:
                raise SystemExit('Models page needs its generated models table.')
            body = body.replace('<p>{{models-table}}</p>', models_table())
        outline='<p class="outline-title">On this page</p>'+''.join(f'<a href="#{item["id"]}">{html.escape(item["name"])}</a>' for item in toc[0].get('children',[]) if item['level']==2) if toc and toc[0].get('children') else ''
        (out/f'{name}.md').write_text(markdown_copy(text, version))
        (out/f'{name}.html').write_text(shell(body,name,title,version,outline=outline,markdown_source=f'{name}.md'))
        titles[name] = title
        search.extend(search_records(body, name, title))
    notebook_pages = []
    for source in sorted((ROOT/'examples/notebooks').glob('*.ipynb')):
        name = 'notebook-' + source.stem
        body, title = notebook_view(source, version)
        (out/f'{name}.html').write_text(shell(body, name, title, version))
        notebook_pages.append(name)
        search.append(dict(title=title, text=f'{title} · Notebook preview · {source.stem}', href=name+'.html'))
    (out/'llms.txt').write_text(llms_index(version, titles))
    (out/'search-index.json').write_text(json.dumps(search,ensure_ascii=False,indent=2)+'\n')
    home=home_body(pages['start'],sdk_version=version,browser_connect=browser_connect_release(version));(out/'index.html').write_text(shell(home,'index','From a DataFrame to a prediction',version,home=True))
    prefix=f'v{version}/';(base/'index.html').write_text(shell(home_body(pages['start'],prefix,sdk_version=version,browser_connect=browser_connect_release(version)),'index','From a DataFrame to a prediction',version,prefix=prefix,home=True))
    (base/'.nojekyll').write_text('')
    urls=[PUBLIC+'/',PUBLIC+f'/v{version}/']+[PUBLIC+f'/v{version}/{name}.html' for name in [*pages, *notebook_pages]]
    urls += [PUBLIC + '/' + name + '.html' for name in [*pages, *notebook_pages]]
    historical=[old for old in older_versions(version) if (base/f'v{old}'/'index.html').exists()]
    for old in historical:urls += [PUBLIC+f'/v{old}/']+[PUBLIC+f'/v{old}/'+p.name for p in sorted((base/f'v{old}').glob('*.html'))]
    (base/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join(f'<url><loc>{url}</loc></url>' for url in urls)+'</urlset>')
    (base/'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: '+PUBLIC+'/sitemap.xml\n')
    manifest=dict(sdk_version=version,pages=sorted(pages),source_tag='v'+version,supported_notebooks=sorted(p.stem for p in downloads.glob('*.ipynb')),historical_versions=sorted(historical,key=release_tuple),scope='bounded qualified preview; full-model/SDK closure gates pending')
    manifest['documentation_ref'] = os.environ.get('DOCS_NOTEBOOK_REF') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest['notebook_views'] = sorted(notebook_pages)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Current, unversioned routes share the same reviewed documentation bytes.
    shutil.copytree(out, base, dirs_exist_ok=True)
    for path in base.glob('*.html'):
        text = path.read_text().replace(f'{PUBLIC}/v{version}/{path.name}', f'{PUBLIC}/{path.name}')
        text = text.replace('href="../v', 'href="v').replace('value="../v', 'value="v')
        path.write_text(text)
    for path in (base/'v0.8.0').glob('*.html'):
        if (base/path.name).exists():
            path.write_text(current_redirect(path.name))
    if (base/'v0.8.0/downloads').exists():
        for source in downloads.iterdir():
            if source.is_file() and source.name != 'forecasting.py':
                shutil.copy2(source, base/'v0.8.0/downloads'/source.name)
    print(f'Built {len(pages)} guides plus landing for SDK {version}; no credentials or live model execution')


if __name__=='__main__':main()
