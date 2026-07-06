from datetime import datetime


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def next_model_version(existing_versions: list[dict]) -> int:
    if not existing_versions:
        return 1
    return max(int(item.get("version", 0)) for item in existing_versions) + 1
