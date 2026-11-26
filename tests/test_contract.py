"""Tests for static contract validation (AST-based, no imports of model.py)."""

import textwrap

import pytest

from mlt.core.contract import Level, check_project, validate_contract
from mlt.core.errors import ContractError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(
    tmp_path,
    *,
    model_py: str | None = None,
    requirements: bool = True,
    params_yaml: str | None = None,
    schema_yaml: str | None = None,
) -> object:
    if model_py is not None:
        (tmp_path / "model.py").write_text(textwrap.dedent(model_py))
    if requirements:
        (tmp_path / "requirements.txt").write_text("scikit-learn>=1.3\n")
    if params_yaml is not None:
        (tmp_path / "params.yaml").write_text(textwrap.dedent(params_yaml))
    if schema_yaml is not None:
        (tmp_path / "schema.yaml").write_text(textwrap.dedent(schema_yaml))
    return tmp_path


def _errors(results):
    return [r for r in results if r.level == Level.ERROR]


def _warnings(results):
    return [r for r in results if r.level == Level.WARNING]


# ---------------------------------------------------------------------------
# Valid projects
# ---------------------------------------------------------------------------

_VALID_SKLEARN = """\
    from mlt.model.sklearn import SklearnModel

    class Model(SklearnModel):
        def fit(self, data_dir, params):
            return {"accuracy": 1.0}
"""

_VALID_MLMODEL = """\
    from mlt.model.base import MLModel

    class Model(MLModel):
        def fit(self, data_dir, params):
            return {"accuracy": 1.0}
        def predict(self, inputs):
            return []
        def save(self, path):
            pass
        @classmethod
        def load(cls, path):
            return cls()
"""


def test_valid_sklearn_project(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN)
    assert _errors(check_project(project)) == []


def test_valid_mlmodel_project(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_MLMODEL)
    assert _errors(check_project(project)) == []


def test_valid_project_with_params_and_schema(tmp_path):
    params = "n_estimators: 100\nmax_depth: 5\n"
    schema = "features:\n  age: {type: int}\n  income: {type: float}\n"
    project = _make_project(
        tmp_path, model_py=_VALID_SKLEARN, params_yaml=params, schema_yaml=schema
    )
    assert _errors(check_project(project)) == []


def test_import_alias_is_recognised(tmp_path):
    """``from mlt.model.sklearn import SklearnModel as Base`` must be accepted."""
    model_py = """\
        from mlt.model.sklearn import SklearnModel as Base

        class Model(Base):
            def fit(self, data_dir, params):
                return {}
    """
    project = _make_project(tmp_path, model_py=model_py)
    assert _errors(check_project(project)) == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_model_py(tmp_path):
    project = _make_project(tmp_path)  # no model_py
    results = check_project(project)
    assert any("model.py not found" in r.message for r in _errors(results))


def test_missing_requirements(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, requirements=False)
    results = check_project(project)
    assert any("requirements.txt" in r.message for r in _errors(results))


def test_no_subclass(tmp_path):
    model_py = "class Model:\n    pass\n"
    project = _make_project(tmp_path, model_py=model_py)
    results = check_project(project)
    assert any("no class" in r.message for r in _errors(results))


def test_multiple_subclasses(tmp_path):
    model_py = """\
        from mlt.model.sklearn import SklearnModel

        class ModelA(SklearnModel):
            def fit(self, data_dir, params):
                return {}

        class ModelB(SklearnModel):
            def fit(self, data_dir, params):
                return {}
    """
    project = _make_project(tmp_path, model_py=model_py)
    results = check_project(project)
    assert any("exactly one" in r.message for r in _errors(results))


def test_missing_fit_in_sklearn_subclass(tmp_path):
    """SklearnModel subclass without fit must fail."""
    model_py = """\
        from mlt.model.sklearn import SklearnModel

        class Model(SklearnModel):
            pass
    """
    project = _make_project(tmp_path, model_py=model_py)
    results = check_project(project)
    assert any("'fit'" in r.message for r in _errors(results))


def test_missing_methods_in_mlmodel_subclass(tmp_path):
    """MLModel subclass missing predict and save must report both."""
    model_py = """\
        from mlt.model.base import MLModel

        class Model(MLModel):
            def fit(self, data_dir, params):
                return {}
            @classmethod
            def load(cls, path):
                return cls()
    """
    project = _make_project(tmp_path, model_py=model_py)
    results = check_project(project)
    missing = {r.message for r in _errors(results) if "must define method" in r.message}
    assert any("'predict'" in m for m in missing)
    assert any("'save'" in m for m in missing)


def test_syntax_error_in_model_py(tmp_path):
    project = _make_project(tmp_path, model_py="def broken(\n")
    results = check_project(project)
    assert any("syntax error" in r.message for r in _errors(results))


def test_invalid_params_yaml_not_a_dict(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, params_yaml="- item1\n- item2\n")
    results = check_project(project)
    assert any("params.yaml" in r.message for r in _errors(results))


def test_invalid_params_yaml_bad_syntax(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, params_yaml="key: [unclosed\n")
    results = check_project(project)
    assert any("params.yaml" in r.message for r in _errors(results))


def test_invalid_schema_yaml_missing_features(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, schema_yaml="something: else\n")
    results = check_project(project)
    assert any("schema.yaml" in r.message for r in _errors(results))


def test_invalid_schema_yaml_features_not_a_dict(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, schema_yaml="features:\n  - col1\n")
    results = check_project(project)
    assert any("schema.yaml" in r.message for r in _errors(results))


# ---------------------------------------------------------------------------
# Indirect inheritance → warning, not error
# ---------------------------------------------------------------------------


def test_indirect_inheritance_produces_warning(tmp_path):
    """Class with fit but unknown base → warning only, not error."""
    model_py = """\
        from some_library import ExternalBase

        class Model(ExternalBase):
            def fit(self, data_dir, params):
                return {}
            def predict(self, inputs):
                return []
            def save(self, path):
                pass
            @classmethod
            def load(cls, path):
                return cls()
    """
    project = _make_project(tmp_path, model_py=model_py)
    results = check_project(project)
    assert _errors(results) == []
    assert len(_warnings(results)) > 0
    assert any("indirect" in r.message for r in _warnings(results))


# ---------------------------------------------------------------------------
# validate_contract raises ContractError
# ---------------------------------------------------------------------------


def test_validate_contract_raises_on_missing_requirements(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN, requirements=False)
    with pytest.raises(ContractError):
        validate_contract(project)


def test_validate_contract_passes_on_valid_project(tmp_path):
    project = _make_project(tmp_path, model_py=_VALID_SKLEARN)
    validate_contract(project)  # must not raise


# ---------------------------------------------------------------------------
# Runner selection (unit test, no venv build)
# ---------------------------------------------------------------------------


def test_auto_runner_falls_back_to_venv_when_docker_unavailable(tmp_path, monkeypatch):
    """runner='auto' must use VenvRunner when Docker is not available."""
    from mlt.core.trainer import _create_runner
    from mlt.runners.venv import VenvRunner

    monkeypatch.setattr("mlt.core.docker_check.docker_available", lambda: False)
    req = tmp_path / "requirements.txt"
    req.write_text("scikit-learn>=1.3\n")
    runner = _create_runner("auto", req, "3.10")
    assert isinstance(runner, VenvRunner)
