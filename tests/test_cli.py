from unittest.mock import MagicMock, patch

import click
import pytest
import torch.nn as nn
from click.testing import CliRunner

from torch_audit.cli import load_model_from_string, main
from torch_audit.core import AuditResult, Severity


# --- Mocks ---
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()


# --- Loader Tests ---


def test_load_model_valid_class():
    # Patch importlib specifically within the cli module namespace
    with patch("torch_audit.cli.importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.MyModel = MockModel
        mock_import.return_value = mock_module

        model = load_model_from_string("pkg:MyModel")
        assert isinstance(model, MockModel)


def test_load_model_import_error():
    with patch("torch_audit.cli.importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError("No module named 'fake'")

        with pytest.raises(click.BadParameter, match="Could not load"):
            load_model_from_string("fake:Model")


def test_load_model_instantiation_failure():
    with patch("torch_audit.cli.importlib.import_module") as mock_import:
        mock_module = MagicMock()

        class Broken(nn.Module):
            def __init__(self):
                raise ValueError("Init failed")

        mock_module.Broken = Broken
        mock_import.return_value = mock_module

        with pytest.raises(click.BadParameter, match="Could not instantiate"):
            load_model_from_string("pkg:Broken")


# --- Main Execution Tests ---


@pytest.fixture
def mock_audit():
    with patch("torch_audit.cli.audit") as mock:
        yield mock


def test_cli_success(mock_audit):
    mock_audit.return_value = AuditResult([], 0, Severity.INFO)

    with patch("torch_audit.cli.load_model_from_string") as mock_loader:
        mock_loader.return_value = MockModel()

        runner = CliRunner()
        result = runner.invoke(main, ["dummy:Model"])

        assert result.exit_code == 0
        mock_audit.assert_called_once()


def test_cli_failure_exit_code(mock_audit):
    mock_audit.return_value = AuditResult([], 1, Severity.ERROR)

    with patch("torch_audit.cli.load_model_from_string") as mock_loader:
        mock_loader.return_value = MockModel()

        runner = CliRunner()
        result = runner.invoke(main, ["dummy:Model"])

        assert result.exit_code == 1
