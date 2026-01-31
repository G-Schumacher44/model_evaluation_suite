"""Unit tests for Click-based CLI."""

from click.testing import CliRunner

from model_eval_suite.cli import cli


class TestCLICommands:
    """Test CLI command invocations."""

    def test_cli_help(self):
        """Test CLI help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Model Evaluation Suite" in result.output
        assert "run" in result.output
        assert "init" in result.output

    def test_cli_version(self):
        """Test CLI version flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_run_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()
        assert "CONFIG_PATH" in result.output

    def test_init_help(self):
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert result.exit_code == 0
        assert "configuration" in result.output.lower()
        assert "--task" in result.output

    def test_prep_help(self):
        """Test prep command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prep", "--help"])

        assert result.exit_code == 0
        assert "dataset" in result.output.lower()

    def test_validate_help(self):
        """Test validate command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "champion" in result.output.lower()

    def test_list_models(self):
        """Test list-models command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-models"])

        assert result.exit_code == 0
        assert "Classification" in result.output
        assert "Regression" in result.output
        assert "XGBoost" in result.output
        assert "LogisticRegression" in result.output


class TestInitCommand:
    """Test init command functionality."""

    def test_init_classification_minimal(self, tmp_path):
        """Test generating minimal classification config."""
        runner = CliRunner()
        output_path = tmp_path / "test_config.yaml"

        result = runner.invoke(
            cli,
            [
                "init",
                "--task",
                "classification",
                "--output",
                str(output_path),
                "--template",
                "minimal",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

        content = output_path.read_text()
        assert "classification" in content
        assert "LogisticRegression" in content
        assert "run_id" in content

    def test_init_regression_minimal(self, tmp_path):
        """Test generating minimal regression config."""
        runner = CliRunner()
        output_path = tmp_path / "test_reg.yaml"

        result = runner.invoke(
            cli,
            ["init", "--task", "regression", "--output", str(output_path), "--template", "minimal"],
        )

        assert result.exit_code == 0
        assert output_path.exists()

        content = output_path.read_text()
        assert "regression" in content
        assert "LinearRegression" in content

    def test_init_creates_parent_dirs(self, tmp_path):
        """Test that init creates parent directories."""
        runner = CliRunner()
        output_path = tmp_path / "nested" / "dir" / "config.yaml"

        result = runner.invoke(
            cli, ["init", "--task", "classification", "--output", str(output_path)]
        )

        assert result.exit_code == 0
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_init_interactive_prompt(self, tmp_path):
        """Test interactive task prompt."""
        runner = CliRunner()
        output_path = tmp_path / "interactive.yaml"

        # Simulate user input: choose 'classification'
        result = runner.invoke(
            cli, ["init", "--output", str(output_path)], input="classification\n"
        )

        assert result.exit_code == 0
        assert output_path.exists()


class TestRunCommand:
    """Test run command functionality."""

    def test_run_missing_config(self):
        """Test run with non-existent config file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "error" in result.output.lower()

    def test_run_with_logging_option(self, tmp_path):
        """Test run with logging option."""
        # Create a dummy config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text("run_id: test\ntask_type: classification\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(config_path), "--logging", "off"])

        # Will fail due to incomplete config, but should accept the flag
        assert "--logging" not in result.output  # Flag was parsed

    def test_run_notebook_mode_flag(self, tmp_path):
        """Test run with notebook mode flag."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("run_id: test\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(config_path), "--notebook-mode"])

        # Should accept the flag even if pipeline fails
        assert result.exit_code in [0, 1]  # May fail on incomplete config


class TestPrepCommand:
    """Test prep command functionality."""

    def test_prep_missing_config(self):
        """Test prep with non-existent config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prep", "nonexistent.yaml"])

        assert result.exit_code != 0


class TestValidateCommand:
    """Test validate command functionality."""

    def test_validate_missing_config(self):
        """Test validate with non-existent config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "nonexistent.yaml"])

        assert result.exit_code != 0

    def test_validate_with_model_options(self, tmp_path):
        """Test validate with model name and version options."""
        config_path = tmp_path / "val_config.yaml"
        config_path.write_text("run_id: test_val\n")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["validate", str(config_path), "--model-name", "test_model", "--version", "1"]
        )

        # Should accept the options even if validation fails
        assert "--model-name" not in result.output  # Options were parsed
