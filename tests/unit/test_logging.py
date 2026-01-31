"""Unit tests for logging configuration."""

import logging

from model_eval_suite.utils.logging import configure_logging


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging_notebook_mode_off(self, temp_dir):
        """Test logging configuration with notebook mode off."""
        log_file = temp_dir / "test.log"

        configure_logging(notebook_mode=False, logging_mode="on", log_path=log_file)

        logger = logging.getLogger()
        assert logger.level == logging.INFO

    def test_configure_logging_creates_log_file(self, temp_dir):
        """Test that log file is created."""
        log_file = temp_dir / "logs" / "test.log"
        log_file.parent.mkdir(exist_ok=True)

        configure_logging(notebook_mode=False, logging_mode="on", log_path=log_file)

        # Log a message
        logging.info("Test message")

        assert log_file.exists()

    def test_logging_mode_off_sets_critical_level(self, temp_dir):
        """Test that logging mode 'off' sets critical level."""
        log_file = temp_dir / "test.log"

        configure_logging(notebook_mode=False, logging_mode="off", log_path=log_file)

        logger = logging.getLogger()
        # Should be CRITICAL + 1 to suppress all logs
        assert logger.level > logging.CRITICAL

    def test_logging_auto_mode_with_notebook(self, temp_dir):
        """Test auto mode with notebook enabled."""
        log_file = temp_dir / "test.log"

        configure_logging(notebook_mode=True, logging_mode="auto", log_path=log_file)

        logger = logging.getLogger()
        # In notebook mode with auto, should suppress console logs
        assert logger.level > logging.INFO or logger.level == logging.WARNING

    def test_third_party_loggers_suppressed(self, temp_dir):
        """Test that third-party loggers are properly suppressed."""
        log_file = temp_dir / "test.log"

        configure_logging(notebook_mode=False, logging_mode="on", log_path=log_file)

        # Check that alembic and mlflow loggers are set to WARNING
        alembic_logger = logging.getLogger("alembic")
        mlflow_logger = logging.getLogger("mlflow.store.db.utils")

        assert alembic_logger.level >= logging.WARNING
        assert mlflow_logger.level >= logging.WARNING
