import importlib


def test_package_imports_resolve_cleanly():
    """
    Ensure all modules are discoverable natively without a root data/ directory.
    This prevents `.gitignore` bugs where `packages/data/` was accidentally ignored.
    """
    # The required modules per the prompt
    modules = [
        "faulttrace_data.benchmarks.scifact",
        "faulttrace_data.benchmarks.hotpotqa",
        "faulttrace_data.benchmarks.covidqa",
        "faulttrace_data.text.analytics"
    ]

    for mod in modules:
        try:
            # We use importlib to ensure they are actually importable
            importlib.import_module(mod)
        except ImportError as e:
            assert False, f"Failed to import {mod}. Ensure __init__.py exists and it is not ignored: {e}"

def test_classes_exist():
    from faulttrace_data.benchmarks.covidqa import CovidQAAdapter
    from faulttrace_data.benchmarks.hotpotqa import HotpotQAAdapter
    from faulttrace_data.benchmarks.scifact import SciFactAdapter
    from faulttrace_data.text.analytics import CorpusAnalytics

    assert SciFactAdapter is not None
    assert HotpotQAAdapter is not None
    assert CovidQAAdapter is not None
    assert CorpusAnalytics is not None
