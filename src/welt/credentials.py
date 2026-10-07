"""Opaque credentials: safe display/serialization, explicit runtime access only."""

import hmac


class Credential:
    def __init__(self, value=None):
        if value is not None and not isinstance(value, str):
            raise ValueError("Credentials must be strings or None.")
        self._value = value

    def get_secret_value(self):
        return self._value

    def __repr__(self):
        return "Credential(<redacted>)"

    __str__ = __repr__

    def __eq__(self, other):
        value = other._value if isinstance(other, Credential) else other
        if self._value is None or value is None:
            return self._value is value
        return isinstance(value, str) and hmac.compare_digest(self._value.encode(), value.encode())

    def __deepcopy__(self, memo):
        return type(self)(self._value)

    def __reduce__(self):
        return (type(self), ())


def credential(value):
    return value if isinstance(value, Credential) or value is None else Credential(value)


def secret(value):
    return value.get_secret_value() if isinstance(value, Credential) else value
