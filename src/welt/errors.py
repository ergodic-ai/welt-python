"""Stable remote errors with actionable metadata."""


class WeltError(Exception):
    def __init__(self, message, *, code="unknown", request_id=None, retryable=False):
        super().__init__(message)
        self.code, self.request_id, self.retryable = code, request_id, retryable


class AuthenticationError(WeltError):
    pass


class InvalidInputError(WeltError, ValueError):
    pass


class ModelUnavailableError(WeltError):
    pass


class CapacityError(WeltError):
    pass


class ExecutionError(WeltError):
    pass


class PredictionPendingError(WeltError):
    def __init__(self, message, *, job_id=None, **kwargs):
        self.job_id = job_id
        super().__init__(message, **kwargs)
