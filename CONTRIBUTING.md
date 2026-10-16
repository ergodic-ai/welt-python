# Contributing

SDK edits belong in this repository. The private service integration pins an
immutable SDK commit/artifact and runs additional real API and account/tenant tests.
Update the public controlled-transport tests and guide for behavior changes. Public
CI must remain independent of private backend imports/resources.

Use Python 3.11–3.13. The dependency lock records the development resolution;
broader sklearn compatibility is an explicit qualification task, not a claim from
a single environment. Keep credentials, data rows from customers, saved prediction
outputs and fitted contexts out of Git. Examples use inputs supplied by the user
or generated synthetic fixtures; no automatic live tests.

Reviewers should check schema/row/class order, job timeout/retry semantics,
credential handling and safe errors. Explicit credentials are redacted in estimator/Pipeline representations,
and pickle strips them for environment-based reauthentication. Retain tests for
these guarantees and keep secrets out of examples regardless of SDK safeguards. Do not invent support for unavailable model capabilities.

Guides/notebooks evolve only with actual released contracts. Build a wheel, install
it into the locked development environment, execute scripts/check_notebooks.py,
and build scripts/build_docs.py. Never commit execution outputs. Versioned docs
publication uses the same public repository; no live CI credentials or FM execution.

## Documentation pages

Pages share the Welt console's Graphite design tokens (`docs/assets/tokens.css`,
copied unchanged with its hash in `docs/assets/design-provenance.json`) and
self-hosted Geist fonts. Style with tokens only; pages make no third-party requests.
Guides are Markdown in `docs/` and can use these components (DEC-073):

- Numbered steps: end consecutive `##` headings with `{.step}`.
- Callouts: `!!! note "Title"`, `!!! tip`, `!!! caution` and `!!! need "You'll need"`,
  with the body indented four spaces.
- Code: ```` ```python file="first.py" hl="3-4" ```` labels a file and highlights lines;
  adjacent fences with `tab="pip"` / `tab="uv"` form one tab group; ```` ```output ````
  directly after code shows its expected output (use `title="…"` to label a recorded run).
- Folded detail: `<details markdown="1"><summary>Question?</summary> … </details>`.
- Next-step cards: wrap a list of links in `<div class="cards" markdown="1"> … </div>`.

Renaming a heading changes its anchor; keep old anchors that published pages or
redirects use. The Models table renders `docs/models-snapshot.json`; refresh it with
`scripts/snapshot_models.py` (a credential-free read of the public catalogue) and
review the diff before committing. The build never calls the live service. Forecasting
profiles (`univariate_forecasting`, `forecasting`) stay in the snapshot but are not rows of the
table: their measured corners are documented in the forecasting guide.
