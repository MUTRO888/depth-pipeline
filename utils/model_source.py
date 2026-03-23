import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return bool(value)


def read_offline_mode(runtime_config=None):
    env_value = os.getenv("DEPTH_PIPELINE_OFFLINE")
    if env_value is not None:
        return env_value.strip().lower() in _TRUE_VALUES

    if not runtime_config:
        return False

    return _as_bool(runtime_config.get("offline", False))


def resolve_path(project_dir, value):
    if not value or not isinstance(value, str):
        return None

    expanded = os.path.expandvars(os.path.expanduser(value))
    if os.path.isabs(expanded):
        return expanded

    return os.path.abspath(os.path.join(project_dir, expanded))


def resolve_existing_path(project_dir, value):
    resolved = resolve_path(project_dir, value)
    if resolved and os.path.exists(resolved):
        return resolved
    return None


def resolve_model_source(
    section,
    project_dir,
    path_keys=("local_path", "path"),
    repo_keys=("name",),
    fallback_keys=(),
    offline=False,
    label="模型",
):
    for key in list(path_keys) + list(fallback_keys):
        value = section.get(key)
        if not value:
            continue

        resolved = resolve_existing_path(project_dir, value)
        if resolved:
            return resolved

    if offline:
        for key in path_keys:
            value = section.get(key)
            if value:
                raise FileNotFoundError(
                    f"离线模式下未找到本地{label}: {resolve_path(project_dir, value)}"
                )

        raise FileNotFoundError(f"离线模式下未配置本地{label}路径")

    for key in repo_keys:
        value = section.get(key)
        if value:
            return value

    for key in fallback_keys:
        value = section.get(key)
        if value:
            return value

    raise FileNotFoundError(f"未配置{label}")
