"""Static contract validation for mlt projects.

Validates ``model.py`` via AST — never imports it — so the scientist's
dependencies do not need to be installed on the host.

Rules enforced here (exit 3 on failure):
  - ``model.py`` exists and parses cleanly
  - ``requirements.txt`` exists
  - Exactly one class directly inheriting from ``MLModel`` or ``SklearnModel``
    (or an import alias of either) is defined
  - That class defines the required methods:
      - ``MLModel`` subclass  → fit, predict, save, load
      - ``SklearnModel`` subclass → fit  (others are provided by the base)
  - ``params.yaml``, if present, is valid YAML and a mapping at the top level
  - ``schema.yaml``, if present, is valid YAML and contains a ``features`` mapping

Indirect inheritance (the base class is itself not from mlt.model) produces
a warning, not an error; ``mlt check --deep`` validates inside the environment.
"""

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Names the tool recognises as direct mlt base classes.
_SKLEARN_BASES = {"SklearnModel"}
_MLMODEL_BASES = {"MLModel"}
_KNOWN_BASES = _MLMODEL_BASES | _SKLEARN_BASES

# Methods required depending on the base class used.
_REQUIRED_FOR_MLMODEL = {"fit", "predict", "save", "load"}
_REQUIRED_FOR_SKLEARN = {"fit"}


class Level(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    level: Level
    message: str
    file: str | None = None
    line: int | None = None
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Return a mapping {local_name: canonical_name} for mlt base class imports.

    Handles both ``from mlt.model import MLModel as Base`` and plain imports.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _KNOWN_BASES:
                    local = alias.asname if alias.asname else alias.name
                    aliases[local] = alias.name
    return aliases


def _base_name(base_node: ast.expr) -> str | None:
    """Extract the name from a base class expression (Name or Attribute)."""
    if isinstance(base_node, ast.Name):
        return base_node.id
    if isinstance(base_node, ast.Attribute):
        return base_node.attr
    return None


def _direct_subclasses(
    tree: ast.Module, known_local_names: set[str]
) -> list[ast.ClassDef]:
    """Find class definitions whose base is in ``known_local_names``."""
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = _base_name(base)
                if name and name in known_local_names:
                    result.append(node)
                    break
    return result


def _defined_methods(cls_node: ast.ClassDef) -> set[str]:
    """Return method names defined directly in the class body (not inherited)."""
    return {
        node.name
        for node in cls_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_model_like_class(tree: ast.Module, known_local_names: set[str]) -> bool:
    """Return True if any class has a ``fit`` method but no recognised base.

    Used to detect indirect inheritance and emit a warning instead of an error.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_direct = any(
                _base_name(b) in known_local_names for b in node.bases
            )
            if is_direct:
                continue
            methods = _defined_methods(node)
            if "fit" in methods:
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_project(project_dir: Path) -> list[CheckResult]:
    """Run all static checks and return a list of results.

    Does not raise; callers inspect the Level of each result.
    """
    results: list[CheckResult] = []

    # -- requirements.txt -------------------------------------------------------
    req = project_dir / "requirements.txt"
    if not req.exists():
        results.append(
            CheckResult(
                level=Level.ERROR,
                message="requirements.txt not found",
                file="requirements.txt",
                suggestion="Create requirements.txt with all dependencies (versioned).",
            )
        )
    else:
        results.append(CheckResult(
            level=Level.OK, message="requirements.txt found", file="requirements.txt"
        ))

    # -- model.py ---------------------------------------------------------------
    model_py = project_dir / "model.py"
    if not model_py.exists():
        results.append(
            CheckResult(
                level=Level.ERROR,
                message="model.py not found",
                file="model.py",
                suggestion="Create model.py with a class inheriting from MLModel or SklearnModel.",
            )
        )
        return results  # nothing else to check without model.py

    try:
        source = model_py.read_text(encoding="utf-8")
        tree = ast.parse(source, filename="model.py")
    except SyntaxError as exc:
        results.append(
            CheckResult(
                level=Level.ERROR,
                message=f"syntax error in model.py: {exc.msg}",
                file="model.py",
                line=exc.lineno,
            )
        )
        return results

    # Resolve import aliases so ``from mlt.model import MLModel as Base`` works.
    alias_map = _collect_import_aliases(tree)
    # local names that are known mlt bases
    known_locals = set(alias_map.keys()) | _KNOWN_BASES

    subclasses = _direct_subclasses(tree, known_locals)

    if len(subclasses) == 0:
        if _has_model_like_class(tree, known_locals):
            results.append(
                CheckResult(
                    level=Level.WARNING,
                    message=(
                        "could not confirm MLModel inheritance (indirect or external base); "
                        "use `mlt check --deep` to validate inside the environment"
                    ),
                    file="model.py",
                    suggestion="Run `mlt check --deep` for full validation.",
                )
            )
        else:
            results.append(
                CheckResult(
                    level=Level.ERROR,
                    message="no class inheriting from MLModel or SklearnModel found in model.py",
                    file="model.py",
                    suggestion="Define a class that inherits from MLModel or SklearnModel.",
                )
            )
        return results

    if len(subclasses) > 1:
        names = ", ".join(c.name for c in subclasses)
        results.append(
            CheckResult(
                level=Level.ERROR,
                message=(
                    f"model.py must define exactly one MLModel subclass, "
                    f"found {len(subclasses)}: {names}"
                ),
                file="model.py",
                suggestion="Keep only one class inheriting from MLModel or SklearnModel.",
            )
        )
        return results

    cls_node = subclasses[0]

    # Determine which base is used to decide required methods.
    base_names_used = {
        _base_name(b)
        for b in cls_node.bases
        if _base_name(b) in known_locals
    }
    # Translate aliases back to canonical names.
    canonical_bases = {alias_map.get(n, n) for n in base_names_used}

    if canonical_bases & _SKLEARN_BASES:
        required = _REQUIRED_FOR_SKLEARN
    else:
        required = _REQUIRED_FOR_MLMODEL

    defined = _defined_methods(cls_node)
    missing = required - defined

    if missing:
        for method in sorted(missing):
            results.append(
                CheckResult(
                    level=Level.ERROR,
                    message=f"class {cls_node.name} must define method '{method}'",
                    file="model.py",
                    line=cls_node.lineno,
                    suggestion=f"Add a '{method}' method to {cls_node.name}.",
                )
            )
    else:
        results.append(
            CheckResult(
                level=Level.OK,
                message=f"class {cls_node.name}: required methods present",
                file="model.py",
                line=cls_node.lineno,
            )
        )

    # -- params.yaml (optional) -------------------------------------------------
    params_yaml = project_dir / "params.yaml"
    if params_yaml.exists():
        try:
            import yaml

            content = yaml.safe_load(params_yaml.read_text(encoding="utf-8"))
            if content is not None and not isinstance(content, dict):
                results.append(
                    CheckResult(
                        level=Level.ERROR,
                        message="params.yaml must be a YAML mapping (dict) at the top level",
                        file="params.yaml",
                        suggestion="Ensure params.yaml contains key: value pairs.",
                    )
                )
            else:
                results.append(
                    CheckResult(level=Level.OK, message="params.yaml is valid", file="params.yaml")
                )
        except Exception as exc:
            results.append(
                CheckResult(
                    level=Level.ERROR,
                    message=f"params.yaml is not valid YAML: {exc}",
                    file="params.yaml",
                )
            )

    # -- schema.yaml (optional) -------------------------------------------------
    schema_yaml = project_dir / "schema.yaml"
    if schema_yaml.exists():
        try:
            import yaml

            content = yaml.safe_load(schema_yaml.read_text(encoding="utf-8"))
            if not isinstance(content, dict) or "features" not in content:
                results.append(
                    CheckResult(
                        level=Level.ERROR,
                        message="schema.yaml must contain a 'features' mapping",
                        file="schema.yaml",
                        suggestion="Add a 'features' key with feature definitions.",
                    )
                )
            elif not isinstance(content["features"], dict):
                results.append(
                    CheckResult(
                        level=Level.ERROR,
                        message="schema.yaml 'features' must be a mapping",
                        file="schema.yaml",
                    )
                )
            else:
                results.append(
                    CheckResult(level=Level.OK, message="schema.yaml is valid", file="schema.yaml")
                )
        except Exception as exc:
            results.append(
                CheckResult(
                    level=Level.ERROR,
                    message=f"schema.yaml is not valid YAML: {exc}",
                    file="schema.yaml",
                )
            )

    return results


def validate_contract(project_dir: Path) -> None:
    """Run static checks and raise ``ContractError`` on the first group of errors.

    Collects all errors into a single message so the user sees everything at once.
    """
    from mlt.core.errors import ContractError

    results = check_project(project_dir)
    errors = [r for r in results if r.level == Level.ERROR]
    if errors:
        lines = []
        for r in errors:
            loc = f"{r.file}" + (f":{r.line}" if r.line else "")
            lines.append(f"  {loc}: {r.message}")
        raise ContractError("contract errors:\n" + "\n".join(lines))
