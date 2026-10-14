"""Provider settings shared by native launchers; credentials never enter config.

Settings use CLI > process environment > selected dotenv > launcher defaults.
REWARD_ENGINE_CREDENTIALS_FILE selects the dotenv, REWARD_ENGINE_KEY_ENV selects
the key variable, and REWARD_ENGINE_BACKEND_URL selects the API base URL.
The selected key's corresponding *_BASE_URL is also supported (e.g.
YIBU_API_KEY / YIBU_BASE_URL). At execution, the key itself uses environment >
dotenv. No key values are copied into frozen sources, manifests or argv.
"""

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


def add_provider_arguments(parser):
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Private dotenv path; overrides REWARD_ENGINE_CREDENTIALS_FILE",
    )
    parser.add_argument(
        "--key-env",
        help="API key variable; overrides REWARD_ENGINE_KEY_ENV (environment > dotenv)",
    )
    parser.add_argument(
        "--backend-url",
        help="API base URL; overrides environment and dotenv provider URLs",
    )


def provider_arguments_supplied(args):
    return any(
        getattr(args, name, None) is not None
        for name in ("credentials", "key_env", "backend_url")
    )


def _dotenv(path):
    # Missing default .env is legitimate for environment-only credentials.
    return dotenv_values(path, interpolate=False) if path.is_file() else {}


def _key_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Provider key variable must be an environment variable name")
    return value


def resolve_provider(
    *,
    credentials=None,
    key_env=None,
    backend_url=None,
    default_credentials,
    default_key_env="YIBU_API_KEY",
    default_backend_url="https://yibuapi.com/v1",
):
    path = (
        Path(
            credentials
            if credentials is not None
            else os.environ.get("REWARD_ENGINE_CREDENTIALS_FILE", default_credentials)
        )
        .expanduser()
        .resolve()
    )
    if path.exists() and not path.is_file():
        raise ValueError("Provider credentials path must be a file")
    values = _dotenv(path)
    key = _key_name(
        key_env
        if key_env is not None
        else os.environ.get(
            "REWARD_ENGINE_KEY_ENV",
            values.get("REWARD_ENGINE_KEY_ENV") or default_key_env,
        )
    )
    url_names = ["REWARD_ENGINE_BACKEND_URL"]
    if key.endswith("_API_KEY"):
        url_names.append(key.removesuffix("_API_KEY") + "_BASE_URL")
    url = backend_url
    if url is None:
        url = next(
            (
                source[name]
                for source in (os.environ, values)
                for name in url_names
                if source.get(name) is not None
            ),
            None,
        )
    if url is None:
        if key != default_key_env:
            raise ValueError(
                "A custom provider key variable requires an explicit base URL"
            )
        url = default_backend_url
    # Avoid storing credentials embedded in URLs, or silently changing API paths.
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() or ord(c) < 32 for c in url)
        )
        # Validate malformed/out-of-range ports without echoing the URL.
        _ = parsed.port
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(
            "Provider base URL must be HTTP(S), without credentials, query or fragment"
        )
    url = url.rstrip("/")
    if not parsed.path.strip("/"):
        url += "/v1"
    return {
        "credentials_file": str(path),
        "backend_key_env": key,
        "backend_url": url,
    }


def prepare_options(args, *, default_credentials):
    """Resolve launcher defaults before entering either benchmark's prepare."""
    provider = resolve_provider(
        credentials=getattr(args, "credentials", None),
        key_env=getattr(args, "key_env", None),
        backend_url=getattr(args, "backend_url", None),
        default_credentials=default_credentials,
    )
    return {
        "credentials": provider["credentials_file"],
        "key_env": provider["backend_key_env"],
        "backend_url": provider["backend_url"],
    }


def provider_key(config):
    """Read the selected secret only at use time; never attach it to config."""
    name = _key_name(config.get("backend_key_env", "YIBU_API_KEY"))
    key = os.environ.get(name)
    if key is None:
        key = _dotenv(Path(config["credentials_file"])).get(name)
    if not isinstance(key, str) or not key.strip():
        raise ValueError(
            f"Missing provider API key in environment or credentials file: {name}"
        )
    return key


def application(config, **kwargs):
    from reward_engine.adapters.workbuddy.gateway import application as gateway

    return gateway(config, backend_key=provider_key(config), **kwargs)


def main():
    """Gateway subprocess entrypoint with the same credential precedence."""
    import uvicorn

    from reward_engine.adapters.workbuddy.io import read

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    uvicorn.run(
        application(config),
        host="0.0.0.0",
        port=config["gateway_port"],
        access_log=False,
    )


if __name__ == "__main__":
    main()
