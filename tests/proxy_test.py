import os

import pytest

from buzz.proxy import PROXY_ENV_VARS, apply_proxy, resolve_proxy


@pytest.fixture()
def clean_proxy_env():
    """Isolate the proxy environment for the duration of a test"""
    saved = {
        name: os.environ.pop(name, None)
        for name in PROXY_ENV_VARS + ("BUZZ_PROXY", "BUZZ_PROXY_APPLIED")
    }
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class TestProxy:
    def test_empty_setting_leaves_environment_untouched(self, clean_proxy_env):
        os.environ["HTTP_PROXY"] = "http://system-proxy:3128"

        assert apply_proxy("") == ""
        # A proxy configured outside of Buzz must not be wiped
        assert os.environ["HTTP_PROXY"] == "http://system-proxy:3128"

    def test_exported_to_every_network_variable(self, clean_proxy_env):
        assert apply_proxy("http://127.0.0.1:7890") == "http://127.0.0.1:7890"

        for name in PROXY_ENV_VARS:
            assert os.environ[name] == "http://127.0.0.1:7890"

    def test_resolve_prefers_the_environment_override(self, clean_proxy_env):
        os.environ["BUZZ_PROXY"] = "http://override:8080"

        assert resolve_proxy() == "http://override:8080"

    def test_explicit_value_is_remembered_as_override(self, clean_proxy_env):
        apply_proxy("http://127.0.0.1:7890")

        assert os.environ["BUZZ_PROXY"] == "http://127.0.0.1:7890"
        assert resolve_proxy() == "http://127.0.0.1:7890"

    def test_clearing_removes_what_buzz_exported(self, clean_proxy_env):
        apply_proxy("http://127.0.0.1:7890")
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"

        assert apply_proxy("") == ""
        assert "HTTP_PROXY" not in os.environ
        assert "BUZZ_PROXY" not in os.environ

    def test_clearing_keeps_a_foreign_proxy(self, clean_proxy_env):
        os.environ["HTTP_PROXY"] = "http://system-proxy:3128"
        apply_proxy("http://127.0.0.1:7890")
        os.environ["HTTP_PROXY"] = "http://system-proxy:3128"

        apply_proxy("")

        assert os.environ["HTTP_PROXY"] == "http://system-proxy:3128"
