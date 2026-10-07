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
credential handling and safe errors. Current estimators accept api_key as a sklearn
constructor parameter: use environment configuration to avoid exposing secrets in
repr/serialization; stronger credential serialization isolation remains a planned
SDK hardening task. Do not invent support for unavailable model capabilities.
