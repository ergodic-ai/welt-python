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
