"""Build a local, versioned SDK documentation site from the matching installed wheel.

API-068 / OBJ-002 / MS-010. Historical pages are rebuilt separately from their
immutable release source; this builder never relabels an older SDK contract.
"""
from pathlib import Path
import argparse
import html
import inspect
from importlib.metadata import version as installed_version
import json
import re
import shutil
import tomllib

import markdown
import welt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "https://ergodic-ai.github.io/welt-python"
REQUIRED = ('Client', 'Job', 'Classifier', 'Regressor', 'AsyncClient', 'AsyncJob',
            'CausalDiscovery', 'CausalResult')
NAV = [('Start', 'index'), ('Guides', 'own-data'), ('Reference', 'reference'), ('Models', 'capabilities')]
GUIDES = [('Start', [('First prediction', 'start'), ('Notebook examples', 'notebooks')]),
          ('Guides', [('Use your own table', 'own-data'), ('Reuse a predictor', 'reuse'),
                      ('Select a model', 'model-selection'), ('Research datasets', 'research-datasets'),
                      ('Wait and recover', 'async-and-errors'), ('Discover a graph', 'causal-discovery')]),
          ('Explore', [('SDK resources', 'usage'), ('Regression', 'regression'),
                       ('Reference', 'reference'), ('Models and limits', 'capabilities'),
                       ('Install and accounts', 'onboarding'), ('Migrate to 0.7', 'migration')])]
# Every documented method gets a return contract and an actionable error boundary.
METHODS = {
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
 'result': ('Poll until completion within a local waiting budget; never resubmit or implicitly cancel.', 'Supervised fit: predictor dictionary. Predict: result dictionary. Discovery: CausalResult.', 'JobTimeoutError with job_id for reconnect; typed failed/cancelled job errors. Synchronous request phase budgets are cooperative, not a hard wall-clock interrupt.'),
 'cancel': ('Explicitly request remote cancellation of this job.', 'Server job metadata dictionary (await for AsyncJob).', 'Authentication, permissions, NotFoundError or job conflict; cancellation races with completion.'),
 'research_datasets': ('Search the research catalogue without uploading or fitting data.', 'Page dictionary: items, next_cursor, total, catalogue_version, facets.', 'Invalid filters/limit, authentication, permission or transport errors.'),
 'research_dataset': ('Inspect licence, schema, recorded targets/splits and eligible content before download.', 'Full research detail dictionary; content=None means bytes are not eligible.', 'Authentication/permission or NotFoundError; never infer redistribution rights from catalogue presence.'),
 'download_research_dataset': ('Download version-pinned canonical bytes with exact size/SHA verification to a caller-chosen new path.', 'pathlib.Path after atomic verified publication. No automatic extraction or upload.', 'FileExistsError for existing destination/symlink; invalid size/version, integrity or TransportError. Interrupted downloads leave no destination.'),
 'datasets': ('List private workspace datasets visible to the current key.', 'List of dataset dictionaries.', 'Authentication, permission or transport errors.'),
 'predictors': ('List durable fitted predictors in the current workspace.', 'List of predictor dictionaries.', 'Authentication, permission or transport errors.'),
 'jobs': ('List durable jobs in the current workspace.', 'List of job dictionaries.', 'Authentication, permission or transport errors.'),
 'usage': ('Inspect operation, lifecycle and attempt workload metadata; count operation IDs rather than events.', 'List of usage dictionaries. No rates, credits, pricing or prediction payloads.', 'Authentication, permission or transport errors.'),
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


def public_signature(obj):
    signature = inspect.signature(obj)
    return signature.replace(parameters=[parameter for name, parameter in signature.parameters.items()
                                         if not name.startswith("_") and name not in ("self", "cls")])


def reference():
    parts = ['# Python reference', 'Signatures below are inspected from the installed matching SDK wheel. Methods are grouped by their task. [Guides](own-data.md) teach complete workflows; this page describes returns and failures. Async resource methods use the same contracts with `await`, except the regular `job(id)` factory.', '## Authentication and errors', 'Authenticated operations use an explicit constructor key or `WELT_API_KEY`; missing-key guidance is local. Missing, invalid/expired, and insufficient-permission failures remain distinct. Never log credentials or raw row payloads. Errors expose safe `code`, `request_id`, `job_id`, `status_code`, `retryable` and `retry_after` metadata when available. See [waiting and recovery](async-and-errors.md) for typed actions.']
    for name in REQUIRED:
        obj = getattr(welt, name)
        parts += [f'## {name}', f'```python\n{name}{inspect.signature(obj)}\n```']
        if name == 'CausalResult':
            parts.append('Immutable native result. Read `id`, `dataset_id`, `job_id`, `variables`, `edge_marks`, `native_graph_type`, `model_version`, `configuration_version`, `score_reference`, `decoder`, `assumptions` and `diagnostics`. The result retains no HTTP client or credentials. Scores remain a separate authenticated read.')
        for key, method in inspect.getmembers(obj, inspect.isroutine):
            if key.startswith('_') or not method.__module__.startswith('welt'):
                continue
            if key not in METHODS:
                raise SystemExit(f'Missing annotated method contract: {name}.{key}')
            purpose, returns, errors = METHODS[key]
            parts += [f'### {name}.{key}', purpose, f'```python\n{name}.{key}{public_signature(method)}\n```', f'**Returns:** {returns}', f'**Errors and limits:** {errors}']
    parts += ['## Error types', 'All remote errors inherit `WeltError`. Table/schema validation may raise ordinary `ValueError` before upload. Do not catch every error and silently fit again.', '| Type | Next action |', '| --- | --- |']
    types = {'AuthenticationError': 'Configure a missing key locally; replace an invalid, expired or revoked key in Account.', 'PermissionDeniedError': 'Use a key with the required workspace write/fit/predict permission.', 'InvalidInputError': 'Check the named schema, explicit target and task limits.', 'ModelUnavailableError': 'Inspect current task_profiles and worker readiness; no automatic model fallback.', 'CapacityError / RateLimitError': 'Respect retry_after and workspace admission limits; inspect existing work.', 'JobTimeoutError / PredictionPendingError': 'Retain job_id and reconnect; local timeout does not cancel.', 'ExecutionError / JobCancelledError': 'Inspect the durable job and safe request/job IDs; intentional retry needs a new decision.', 'NotFoundError / ConflictError': 'Check workspace ownership and resource dependencies.', 'TransportError': 'Inspect existing job state before replaying a mutation; GET retries are bounded.', 'ResultExpiredError / ResultDeletedError': 'Temporary access expiry and explicit primary deletion are distinct.', 'InvalidCausalResultError': 'Reject malformed native payload without repair.', 'UnsupportedGraphConversionError / OptionalDependencyError': 'Preserve native graph; convert declared DAGs only with approved local co-release dependency.'}
    parts += [f'| `{name}` | {meaning} |' for name, meaning in types.items()]
    return '\n\n'.join(parts[:-len(types)-2]) + '\n\n' + '\n'.join(parts[-len(types)-2:])


def render_markdown(text):
    text = re.sub(r'\]\(([^():#]+)\.md(#[^)]*)?\)', lambda m: '](' + m[1] + '.html' + (m[2] or '') + ')', text)
    renderer = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc'], extension_configs={'toc': {'permalink': False}})
    body = renderer.convert(text)
    # Highlight raw code without executing it or adding a runtime dependency.
    def code(match):
        language = match[1] or 'text'; raw = html.unescape(match[2])
        if language == 'python':
            token = re.compile(r'("[^"\n]*"|\x27[^\x27\n]*\x27|#[^\n]*|\b(?:from|import|with|as|if|else|for|in|not|and|or|None|True|False|await|async|try|except|finally|raise|assert|return|def)\b)')
            pieces=[];cursor=0
            for item in token.finditer(raw):
                pieces.append(html.escape(raw[cursor:item.start()])); value=item[0];kind='co' if value.startswith('#') else 'st' if value.startswith(('"',"'")) else 'kw';pieces.append(f'<span class="{kind}">{html.escape(value)}</span>');cursor=item.end()
            pieces.append(html.escape(raw[cursor:])); highlighted=''.join(pieces)
        else: highlighted=html.escape(raw)
        return f'<div class="code-wrap"><span class="code-language">{html.escape(language)}</span><pre tabindex="0" aria-label="{html.escape(language)} code"><code>{highlighted}</code></pre></div>'
    body = re.sub(r'<pre><code(?: class="language-([^\"]+)")?>(.*?)</code></pre>', code, body, flags=re.S)
    body = body.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable table"><table>').replace('</table>', '</table></div>')
    body = re.sub(r'<h([23]) id="([^"]+)">(.*?)</h\1>', lambda m: f'<h{m[1]} id="{m[2]}">{m[3]}<a class="anchor" href="#{m[2]}" aria-label="Link to {html.escape(re.sub("<[^>]+>", "", m[3]), quote=True)}">#</a></h{m[1]}>', body)
    return body, renderer.toc_tokens


def shell(body, name, title, version, prefix='', outline='', home=False):
    def link(page): return prefix + ('index.html' if page == 'index' else page + '.html')
    group = 'Start' if name in ('index', 'start', 'onboarding', 'notebooks') else 'Reference' if name == 'reference' else 'Models' if name == 'capabilities' else 'Guides'
    nav = ''.join(f'<a href="{link(page)}"' + (' aria-current="page"' if label == group else '') + f'>{label}</a>' for label, page in NAV)
    sidebar = ''.join(f'<p>{label}</p>' + ''.join(f'<a href="{link(page)}"' + (' aria-current="page"' if page == name else '') + f'>{text}</a>' for text,page in pages) for label,pages in GUIDES)
    versions = f'<option value="{prefix}index.html">v{version}</option><option value="{"../" if prefix == "" else ""}v0.6.0/">v0.6.0</option>'
    css = prefix+'assets/docs.css'; js = prefix+'assets/docs.js'
    canonical = PUBLIC + ('/' if home and prefix else f'/v{version}/' + ('index.html' if name == 'index' else name+'.html'))
    main = f'<main id="main" tabindex="-1" class="home">{body}</main>' if home else f'<div class="layout"><aside class="sidebar" aria-label="Guide navigation">{sidebar}</aside><main id="main" tabindex="-1" class="article">{body}</main><aside class="outline" aria-label="On this page">{outline}</aside></div>'
    return f'''<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Welt Python SDK {version}: DataFrame prediction, reusable predictors, research datasets and native causal discovery."><meta name="color-scheme" content="dark light"><title>{html.escape(title)} · Welt Python SDK {version}</title><link rel="canonical" href="{canonical}"><link rel="icon" type="image/svg+xml" href="{prefix}assets/ergodic-mark-dark.svg"><link rel="stylesheet" href="{css}"><script defer src="{js}"></script></head><body data-search="{prefix}search-index.json"><a class="skip" href="#main">Skip to content</a><header class="header"><a class="brand" href="{link('index')}"><img class="logo-dark" src="{prefix}assets/ergodic-mark-dark.svg" alt="" width="30" height="30"><img class="logo-light" src="{prefix}assets/ergodic-mark-light.svg" alt="" width="30" height="30"><strong>Welt</strong><span>Python SDK</span></a><nav class="topnav" aria-label="Main navigation">{nav}</nav><div class="tools"><select id="version" class="version" aria-label="Documentation version">{versions}</select><button id="search" type="button" aria-haspopup="dialog"><span class="search-label">Search</span><span aria-hidden="true"> ⌕</span><span class="sr-only">Search documentation</span></button><button id="theme" type="button">Light</button><button id="menu" class="menu-button" type="button" aria-haspopup="dialog">Menu</button></div></header>{main}<footer class="footer"><span>Welt by Ergodic · Python SDK {version}</span><span><a href="https://welt.ergodic.dev">Workspace ↗</a> &nbsp; <a href="https://github.com/ergodic-ai/welt-python/tree/v{version}">Release source ↗</a></span></footer><dialog id="navigation" class="nav-dialog" aria-labelledby="navigation-title"><header><strong id="navigation-title">Documentation</strong><button type="button" data-close aria-label="Close navigation">Close</button></header><nav aria-label="Mobile guide navigation">{nav}{sidebar}</nav></dialog><dialog id="search-dialog" aria-labelledby="search-title"><header><strong id="search-title">Search SDK {version}</strong><button type="button" data-close aria-label="Close search">Close</button></header><label for="search-input">Task, method or error</label><input id="search-input" type="search" autocomplete="off" placeholder="Try reopen, target or JobTimeoutError"><ul id="search-results"></ul><p id="search-empty">Search this SDK version by task, method or error.</p></dialog><div id="announcement" class="sr-only" role="status" aria-live="polite"></div></body></html>'''


def home_body(start, prefix=''):
    blocks = re.findall(r'```(\w+)\n(.*?)```', start, re.S)
    install = blocks[0][1].splitlines()[-1]
    key = blocks[1][1]
    classification = blocks[2][1]
    regression = blocks[4][1]
    def code(value, lang='python'): return render_markdown(f'```{lang}\n{value}\n```')[0]
    def link(page): return prefix+page+'.html'
    def example(value):
        setup, primary = value.split('model = ', 1)
        full = code('model = ' + primary)
        full = full.replace('<div class="code-wrap">', '<div class="code-wrap" data-copy="' + html.escape(value, quote=True) + '" data-copy-label="Copy complete example">', 1)
        return '<details class="sample-data"><summary>Sample data and imports · included in Copy</summary>' + code(setup.rstrip()) + '</details>' + full
    stages=''.join('<i></i>' for _ in range(15)); bars=''.join(f'<i style="--h:{height}px;--d:{i*.2}s"></i>' for i,height in enumerate((27,49,35,63,42,57)))
    return f'''<div class="hero"><section class="hero-intro"><h1>From a DataFrame<br><span>to a prediction.</span></h1><p class="hero-copy">Your table. A foundation model. A predictor you can reuse.<br>Start with Python you already know.</p><div class="visual" role="img" aria-label="Illustration: your training table becomes a reusable predictor, which returns predictions for new rows."><div class="visual-stages"><span>01 &nbsp; Your table</span><i class="line"></i><span>02 &nbsp; Predictor</span><i class="line"></i><span>03 &nbsp; Predictions</span></div><div class="visual-scene" aria-hidden="true"><div class="visual-table">{stages}</div><div class="predictor-glyph"></div><div class="visual-points">{bars}</div></div></div><p class="visual-note">Illustrative workflow. No request runs in this page.<button id="motion" class="motion-control" type="button" aria-pressed="false">Pause animation</button></p><p class="boundary">Fit uploads your table to your private workspace and waits for remote preparation. Prediction executes remotely. Your fitted predictor stays reusable after this Python session ends.</p><p class="boundary"><a href="{link('capabilities')}">Models and limits ↗</a> &nbsp; <a href="{link('start')}">Full walkthrough ↗</a></p></section><section class="quickstart" aria-labelledby="quickstart-title"><h2 id="quickstart-title">Your first prediction</h2><p class="intro">Python 3.11–3.13 · SDK 0.7.0</p><div class="step"><span>1 &nbsp; Install the release</span><a href="{link('start')}#install">Environment setup ↗</a></div>{code(install,'sh')}<div class="step"><span>2 &nbsp; Create a workspace key</span><a href="https://welt.ergodic.dev">Open Welt ↗</a></div><p class="credential-note">In Account, enable <strong>Allow writes, fits and predictions</strong>. Keep the key in local secret configuration or use the hidden Python prompt below.</p><details><summary>Set your key safely in Python</summary>{code(key)}</details><div class="step"><span>3 &nbsp; Fit and predict</span></div><div class="task-tabs" role="tablist" aria-label="Prediction task"><button id="classification-tab" role="tab" type="button" aria-selected="true" aria-controls="classification-panel">Classification</button><button id="regression-tab" role="tab" type="button" aria-selected="false" aria-controls="regression-panel" tabindex="-1">Regression</button></div><div id="classification-panel" class="task-panel" role="tabpanel" aria-labelledby="classification-tab">{example(classification)}<div class="result"><strong>What you get</strong><code>ndarray (32,)</code> — one class label per query row, in order.<br>Expected type and shape; actual labels come from the model.</div></div><div id="regression-panel" class="task-panel" role="tabpanel" aria-labelledby="regression-tab" hidden>{example(regression)}<div class="result"><strong>What you get</strong><code>ndarray (32,)</code> — one finite point in target units per row.<br>Expected type and shape; no calibrated intervals or accuracy claim.</div></div><p class="credential-note">Created datasets and predictors remain in your workspace. A local timeout leaves work running; <a href="{link('async-and-errors')}">reconnect by job ID</a>.</p></section></div><section class="home-bottom" aria-label="Continue learning"><div><a href="{link('own-data')}">Use your own table ↗</a><p>Declare the target, match the columns, and predict on new rows.</p></div><div><a href="{link('reuse')}">Reopen your predictor ↗</a><p>Keep the fitted identity. Predict again without uploading training data.</p></div><div><a href="{link('notebooks')}">Learn in a notebook ↗</a><p>Clear, runnable examples for prediction, research and native discovery.</p></div></section>'''


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
    shutil.copy2(ROOT/'examples/first_prediction.py',downloads/'first_prediction.py')
    search=[]
    for name,text in pages.items():
        body,toc=render_markdown(text); title=re.search(r'^# (.+)$',text,re.M)[1]
        outline='<strong>On this page</strong>'+''.join(f'<a href="#{item["id"]}">{html.escape(item["name"])}</a>' for item in toc[0].get('children',[]) if item['level']==2) if toc else ''
        (out/f'{name}.html').write_text(shell(body,name,title,version,outline=outline))
        # One search record per section; links and snippets remain scoped to this release.
        sections=re.split(r'(?m)^## ',text)
        for section in sections:
            heading=section.splitlines()[0].lstrip('# ');slug=re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-')
            plain=re.sub(r'[`#*|\[\]()]','',re.sub(r'<[^>]+>','',section));plain=' '.join(plain.split())
            search.append(dict(title=f'{title} · {heading}',text=plain,href=name+'.html'+('#'+slug if section!=sections[0] else '')))
    (out/'search-index.json').write_text(json.dumps(search,ensure_ascii=False,indent=2)+'\n')
    home=home_body(pages['start']);(out/'index.html').write_text(shell(home,'index','From a DataFrame to a prediction',version,home=True))
    prefix=f'v{version}/';(base/'index.html').write_text(shell(home_body(pages['start'],prefix),'index','From a DataFrame to a prediction',version,prefix=prefix,home=True))
    (base/'.nojekyll').write_text('')
    urls=[PUBLIC+'/',PUBLIC+f'/v{version}/']+[PUBLIC+f'/v{version}/{name}.html' for name in pages]
    if (base/'v0.6.0/index.html').exists():urls += [PUBLIC+'/v0.6.0/']+[PUBLIC+'/v0.6.0/'+p.name for p in (base/'v0.6.0').glob('*.html')]
    (base/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join(f'<url><loc>{url}</loc></url>' for url in urls)+'</urlset>')
    (base/'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: '+PUBLIC+'/sitemap.xml\n')
    manifest=dict(sdk_version=version,pages=sorted(pages),source_tag='v'+version,supported_notebooks=sorted(p.stem for p in downloads.glob('*.ipynb')),historical_versions=['0.6.0'] if (base/'v0.6.0/index.html').exists() else [],scope='bounded qualified preview; full-model/SDK closure gates pending')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {len(pages)} guides plus landing for SDK {version}; no credentials or model outputs included')


if __name__=='__main__':main()
