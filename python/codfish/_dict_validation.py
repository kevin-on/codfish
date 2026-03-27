from __future__ import annotations

_MISSING = object()


def require_dict(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool = True,
) -> dict[str, object]:
    value = _lookup(payload, key, context, distinguish_missing=distinguish_missing)
    if not isinstance(value, dict):
        raise ValueError(f"{context} field {key} must be a dict")
    return value


def optional_require_dict(
    payload: dict[str, object], key: str, context: str
) -> dict[str, object] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{context} field {key} must be a dict or None")
    return value


def require_int(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool = True,
) -> int:
    value = _lookup(payload, key, context, distinguish_missing=distinguish_missing)
    if not isinstance(value, int):
        raise ValueError(f"{context} field {key} must be an int")
    return value


def require_str(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool = True,
) -> str:
    value = _lookup(payload, key, context, distinguish_missing=distinguish_missing)
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key} must be a str")
    return value


def optional_require_str(
    payload: dict[str, object], key: str, context: str
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key} must be a str or None")
    return value


def require_real(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool = True,
) -> float:
    value = _lookup(payload, key, context, distinguish_missing=distinguish_missing)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} field {key} must be a real number")
    return float(value)


def require_positive_int(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool = True,
) -> int:
    value = _lookup(payload, key, context, distinguish_missing=distinguish_missing)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} field {key} must be a positive int")
    return value


def _lookup(
    payload: dict[str, object],
    key: str,
    context: str,
    *,
    distinguish_missing: bool,
) -> object:
    if key in payload:
        return payload[key]
    if distinguish_missing:
        raise ValueError(f"{context} is missing field {key}")
    return _MISSING
