# Install, accounts and API keys

Clone this public repository and install `welt-client` from a reviewed source
revision or built wheel. Import `welt`. Python3.11–3.13 is declared; public CI checks
those versions. No PyPI publication is currently claimed.

```sh
python -m pip install .
# Optional Parquet reader:
python -m pip install '.[parquet]'
```

Use your deployed Welt console for signup, managed email verification, signin and
workspace selection. Create a workspace SDK key there and copy it once into your
local secret configuration. Browser console sessions manage accounts/keys; never
persist SDK keys in browser storage or paste them into notebook cells. Verification
and recovery require actual managed email delivery; a reachable form is not proof
of the full released onboarding flow.

`WELT_BASE_URL` is your service origin and `WELT_API_KEY` is your local workspace
credential. Use environment configuration or your secret manager, keeping values
out of Git, repr/logs and notebook outputs. SDK requests are authorized by the key's
current workspace/scope. Revocation denies subsequent access; already accepted
durable jobs continue until explicit cancellation, but their result retrieval still
requires current authorization. Rotate by issuing a new key, switching clients and
revoking the old key. Account/role changes are independent of model identity.

Use a context manager for `Client`, and close explicit estimator submission job
clients when finished. A fitted predictor is reopened by identity with current
credentials; it does not carry portable authentication or shared weights.
