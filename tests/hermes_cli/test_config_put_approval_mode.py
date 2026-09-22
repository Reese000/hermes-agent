"""PUT /api/config must be able to save at all.

_approval_mode_of lazily imports the approvals.mode normalizer; after the
09-03 facade cleanup dropped the tools.approval re-export, the import still
pointed there and EVERY config save (settings Reasoning/Speed/Approvals)
500'd with ImportError. The normalizer lives in tools.approval_context —
this test pins the import to reality.
"""


def test_approval_mode_of_saves_via_current_normalizer_home():
    from hermes_cli.web_server import _approval_mode_of

    # The schema default (DEFAULT_CONFIG["approvals"]["mode"]) is "smart".
    assert _approval_mode_of({}) == "smart"
    # YAML 1.1 parses a bare `mode: off` as False — must normalize to "off".
    assert _approval_mode_of({"approvals": {"mode": False}}) == "off"
    assert _approval_mode_of({"approvals": {"mode": "SMART"}}) == "smart"
