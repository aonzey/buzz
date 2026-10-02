"""Application wide proxy handling.

Buzz talks to several different network stacks (Hugging Face model downloads,
yt-dlp media downloads, the OpenAI compatible translation client). All of them
honour the standard proxy environment variables, so a single place that exports
them is enough to make the whole application use the proxy.
"""

import logging
import os
from typing import Optional

# Both cases are exported: httpx/openai read the upper case names, some
# libraries (for example yt-dlp) only look at the lower case ones.
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)

# Overrides the saved preference; used by the ``--proxy`` command line option.
PROXY_OVERRIDE_ENV_VAR = "BUZZ_PROXY"

# Remembers which value Buzz itself exported so it can undo it later.
APPLIED_ENV_VAR = "BUZZ_PROXY_APPLIED"


def resolve_proxy() -> str:
    """Return the proxy Buzz should use, or "" when none is configured."""
    override = (os.environ.get(PROXY_OVERRIDE_ENV_VAR) or "").strip()
    if override:
        return override

    try:
        from buzz.settings.settings import Settings

        saved = Settings().value(key=Settings.Key.PROXY, default_value="")
    except Exception:
        logging.debug("Could not read the saved proxy setting", exc_info=True)
        return ""

    return (saved or "").strip()


def apply_proxy(proxy: Optional[str] = None) -> str:
    """Export ``proxy`` to the environment so every network stack picks it up.

    Passing ``None`` uses :func:`resolve_proxy`. An empty value clears a proxy
    that Buzz exported earlier and otherwise leaves the environment untouched,
    so a proxy configured outside of Buzz is never wiped by accident.
    """
    if proxy is None:
        proxy = resolve_proxy()
    else:
        proxy = (proxy or "").strip()
        if proxy:
            os.environ[PROXY_OVERRIDE_ENV_VAR] = proxy
        else:
            os.environ.pop(PROXY_OVERRIDE_ENV_VAR, None)

    if not proxy:
        _clear_applied_proxy()
        return ""

    os.environ[APPLIED_ENV_VAR] = proxy
    for name in PROXY_ENV_VARS:
        os.environ[name] = proxy

    logging.debug("Using proxy: %s", proxy)
    return proxy


def _clear_applied_proxy():
    previous = os.environ.pop(APPLIED_ENV_VAR, "")
    if not previous:
        return

    for name in PROXY_ENV_VARS:
        if os.environ.get(name) == previous:
            os.environ.pop(name, None)
