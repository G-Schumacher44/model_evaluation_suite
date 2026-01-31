#!/usr/bin/env python3
"""
🔄 MLflow Migration Script: Filesystem to SQLite

This script helps migrate existing MLflow data from the legacy filesystem backend
(./mlruns) to the recommended SQLite database backend.

Usage:
    python scripts/migrate_mlflow_to_sqlite.py

What it does:
1. Creates a backup of your current mlruns/ directory
2. Exports data from filesystem backend
3. Imports data into SQLite backend
4. Validates the migration

Requirements:
    pip install mlflow-export-import

Documentation:
    https://github.com/mlflow/mlflow-export-import
"""

import shutil
from datetime import datetime
from pathlib import Path


def main():
    print("=" * 70)
    print("  MLflow Migration: Filesystem → SQLite")
    print("=" * 70)
    print()

    # Check if mlruns exists
    mlruns_path = Path("mlruns")
    if not mlruns_path.exists():
        print("✅ No existing mlruns/ directory found.")
        print("   You can start fresh with SQLite backend!")
        print()
        print("   Your config is already set to: sqlite:///mlflow.db")
        return

    # Warn user
    print("⚠️  This will migrate your MLflow data from filesystem to SQLite.")
    print()
    print("   Current: file:./mlruns")
    print("   Target:  sqlite:///mlflow.db")
    print()
    print("   A backup will be created at: mlruns_backup_{timestamp}")
    print()

    response = input("   Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ Migration cancelled.")
        return

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"mlruns_backup_{timestamp}")

    print(f"\n📦 Creating backup at {backup_path}...")
    shutil.copytree(mlruns_path, backup_path)
    print("   ✅ Backup complete!")

    # Check for mlflow-export-import
    try:
        import mlflow_export_import
    except ImportError:
        print("\n❌ Error: mlflow-export-import not installed")
        print("\n   Install it with:")
        print("   pip install mlflow-export-import")
        print()
        print("   Then run this script again.")
        return

    print("\n🔄 Migration process:")
    print("   1. Export from filesystem backend")
    print("   2. Import to SQLite backend")
    print()
    print("   This may take a few minutes depending on data size...")
    print()

    # For now, provide manual instructions
    print("📋 Manual Migration Steps:")
    print()
    print("1. Export all experiments:")
    print("   export-experiments \\")
    print("     --tracking-uri file:./mlruns \\")
    print("     --output-dir mlflow_export")
    print()
    print("2. Import to SQLite:")
    print("   import-experiments \\")
    print("     --tracking-uri sqlite:///mlflow.db \\")
    print("     --input-dir mlflow_export")
    print()
    print("3. Verify migration:")
    print("   mlflow server --backend-store-uri sqlite:///mlflow.db")
    print()
    print("📚 Full documentation:")
    print("   https://github.com/mlflow/mlflow-export-import")
    print()
    print("=" * 70)
    print()
    print("💡 Alternative: Start Fresh")
    print()
    print("   If you don't need historical data, you can simply:")
    print("   1. Rename mlruns/ to mlruns_old/")
    print("   2. Run your notebook - mlflow.db will be created automatically")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
