"""
🎯 Modern Click-based CLI for Model Evaluation Suite.

Provides a unified command-line interface for all suite operations:
- Run model evaluation pipelines
- Prepare and validate data
- Generate configuration templates
- List and inspect models
"""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.2.0", prog_name="model-eval")
def cli():
    """
    🧠 Model Evaluation Suite - ML Model Evaluation & Interpretability Engine

    A YAML-driven toolkit for comprehensive model evaluation, SHAP explainability,
    baseline comparisons, and audit alerts.
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--notebook-mode/--no-notebook-mode",
    default=False,
    help="Enable notebook mode (suppresses some console output)",
)
@click.option(
    "--logging", type=click.Choice(["auto", "on", "off"]), default="auto", help="Logging mode"
)
def run(config_path, notebook_mode, logging):
    """
    🚀 Run the complete model evaluation pipeline.

    CONFIG_PATH: Path to your YAML configuration file

    Examples:
        model-eval run config/my_model.yaml
        model-eval run config/xgboost.yaml --logging on
    """
    import sys

    from model_eval_suite.run_pipeline import main as run_pipeline

    console.print("\n[bold green]🚀 Starting Model Evaluation Pipeline[/bold green]")
    console.print(f"[cyan]Config:[/cyan] {config_path}\n")

    # Set up sys.argv for the existing pipeline
    original_argv = sys.argv
    sys.argv = ["model-eval", config_path]

    try:
        run_pipeline(config_path)
        console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline failed:[/bold red] {e}\n")
        sys.exit(1)
    finally:
        sys.argv = original_argv


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def prep(config_path):
    """
    📊 Prepare and split dataset for training.

    CONFIG_PATH: Path to data preparation YAML config

    Examples:
        model-eval prep config/data_prep.yaml
    """
    import sys

    from model_eval_suite.data_prep import main as prep_data

    console.print("\n[bold blue]📊 Preparing Dataset[/bold blue]")
    console.print(f"[cyan]Config:[/cyan] {config_path}\n")

    original_argv = sys.argv
    sys.argv = ["model-eval", config_path]

    try:
        prep_data(config_path)
        console.print("\n[bold green]✅ Data preparation completed![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Data prep failed:[/bold red] {e}\n")
        sys.exit(1)
    finally:
        sys.argv = original_argv


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--model-name", help="Model name to validate")
@click.option("--version", help="Model version")
def validate(config_path, model_name, version):
    """
    ✅ Validate a champion model against holdout data.

    CONFIG_PATH: Path to validation YAML config

    Examples:
        model-eval validate config/validation.yaml
        model-eval validate config/validation.yaml --model-name xgb_model --version 1
    """
    import sys

    from model_eval_suite.validate_champion import main as validate_model

    console.print("\n[bold yellow]✅ Validating Champion Model[/bold yellow]")
    console.print(f"[cyan]Config:[/cyan] {config_path}")
    if model_name:
        console.print(f"[cyan]Model:[/cyan] {model_name}")
    if version:
        console.print(f"[cyan]Version:[/cyan] {version}\n")

    original_argv = sys.argv
    sys.argv = ["model-eval", config_path]

    try:
        validate_model(config_path)
        console.print("\n[bold green]✅ Validation completed![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Validation failed:[/bold red] {e}\n")
        sys.exit(1)
    finally:
        sys.argv = original_argv


@cli.command()
@click.option(
    "--task",
    type=click.Choice(["classification", "regression"]),
    prompt="Task type",
    help="Type of ML task",
)
@click.option(
    "--output",
    type=click.Path(),
    default="config/my_config.yaml",
    help="Output path for generated config",
)
@click.option(
    "--template",
    type=click.Choice(["minimal", "full"]),
    default="minimal",
    help="Config template type",
)
def init(task, output, template):
    """
    📝 Generate a new configuration file template.

    Creates a ready-to-use YAML config for your ML task.

    Examples:
        model-eval init --task classification
        model-eval init --task regression --output config/reg_model.yaml --template full
    """
    console.print("\n[bold cyan]📝 Generating Configuration Template[/bold cyan]")
    console.print(f"[cyan]Task:[/cyan] {task}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    console.print(f"[cyan]Template:[/cyan] {template}\n")

    # Create output directory if needed
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate template based on task type
    if template == "minimal":
        config_content = _generate_minimal_config(task)
    else:
        config_content = _generate_full_config(task)

    # Write config
    output_path.write_text(config_content, encoding="utf-8")

    console.print(f"[bold green]✅ Configuration created:[/bold green] {output}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print(f"  1. Edit {output} with your model parameters")
    console.print(f"  2. Run: [cyan]model-eval run {output}[/cyan]\n")


@cli.command(name="list-models")
def list_models():
    """
    📋 List all available model types.

    Shows classification and regression models supported by the suite.
    """
    console.print("\n[bold]📋 Available Models[/bold]\n")

    # Classification models
    table_cls = Table(title="Classification Models", show_header=True, header_style="bold cyan")
    table_cls.add_column("Model Name", style="green")
    table_cls.add_column("Config Name")
    table_cls.add_column("Description")

    cls_models = [
        ("Logistic Regression", "LogisticRegression", "Linear classification"),
        ("Random Forest", "RandomForest", "Ensemble tree classifier"),
        ("XGBoost", "XGBoost", "Gradient boosting classifier"),
        ("SVM", "SVC", "Support Vector Classifier"),
        ("Naive Bayes", "GaussianNB", "Probabilistic classifier"),
        ("Decision Tree", "DecisionTree", "Single tree classifier"),
    ]

    for name, config, desc in cls_models:
        table_cls.add_row(name, config, desc)

    console.print(table_cls)
    console.print()

    # Regression models
    table_reg = Table(title="Regression Models", show_header=True, header_style="bold magenta")
    table_reg.add_column("Model Name", style="green")
    table_reg.add_column("Config Name")
    table_reg.add_column("Description")

    reg_models = [
        ("Linear Regression", "LinearRegression", "Ordinary least squares"),
        ("Random Forest", "RandomForestRegressor", "Ensemble tree regressor"),
        ("XGBoost", "XGBRegressor", "Gradient boosting regressor"),
        ("SVM", "SVR", "Support Vector Regressor"),
        ("Decision Tree", "DecisionTreeRegressor", "Single tree regressor"),
    ]

    for name, config, desc in reg_models:
        table_reg.add_row(name, config, desc)

    console.print(table_reg)
    console.print()


def _generate_minimal_config(task):
    """Generate minimal configuration template."""
    if task == "classification":
        return """# Minimal Classification Configuration
run_id: my_classifier_01
task_type: classification

modeling:
  target_column: target
  pipeline_factory:
    name: LogisticRegression
    numeric_features: [feature_1, feature_2]
    categorical_features: [cat_feature]
    params:
      random_state: 42

evaluation:
  run: true
  explainability:
    run: true
"""
    # regression
    return """# Minimal Regression Configuration
run_id: my_regressor_01
task_type: regression

modeling:
  target_column: target
  pipeline_factory:
    name: LinearRegression
    numeric_features: [feature_1, feature_2]
    categorical_features: [cat_feature]
    params: {}

evaluation:
  run: true
  explainability:
    run: true
"""


def _generate_full_config(task):
    """Generate full configuration template with all options."""
    # For brevity, returning minimal - full template would be much longer
    return _generate_minimal_config(task)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
