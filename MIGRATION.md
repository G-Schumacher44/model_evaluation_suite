# 🔄 Migration Guide: MLflow Filesystem → SQLite

## Overview

Starting in **v0.2.0**, the Model Evaluation Suite uses **SQLite** as the default MLflow tracking backend instead of the legacy filesystem backend (`file:./mlruns`).

This change eliminates deprecation warnings and provides better performance and reliability.

---

## What Changed?

### Before (v0.1.0)
- MLflow tracking: `file:./mlruns`
- Default config: No explicit `mlflow_tracking_uri`
- Data stored in: `mlruns/` directory structure

### After (v0.2.0)
- MLflow tracking: `sqlite:///mlflow.db`
- Default config: `mlflow_tracking_uri: "sqlite:///mlflow.db"`
- Data stored in: Single `mlflow.db` SQLite database

---

## Migration Options

You have **three options** depending on your needs:

### Option 1: Start Fresh (Recommended for Most Users)

If you don't need to preserve existing experiment history:

```bash
# Backup old data (optional)
mv mlruns mlruns_old_backup

# That's it! The suite will create mlflow.db automatically on next run
```

### Option 2: Keep Using Filesystem Backend

If you want to continue using the filesystem backend (not recommended):

**In your user config YAML:**
```yaml
mlflow_tracking_uri: "file:./mlruns"
```

**In default_config.yaml:**
```yaml
base_config: &base_config
  mlflow_tracking_uri: "file:./mlruns"  # Override the default
```

> ⚠️ Note: You'll continue to see deprecation warnings until MLflow removes filesystem support (Feb 2026).

### Option 3: Migrate Existing Data

If you have important historical experiments you want to preserve:

**Manual Migration Steps:**

1. **Install migration tool:**
   ```bash
   pip install mlflow-export-import
   ```

2. **Export from filesystem:**
   ```bash
   export-experiments \
     --tracking-uri file:./mlruns \
     --output-dir mlflow_export
   ```

3. **Import to SQLite:**
   ```bash
   import-experiments \
     --tracking-uri sqlite:///mlflow.db \
     --input-dir mlflow_export
   ```

4. **Verify migration:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
   ```

   Open http://localhost:5000 to verify your experiments are present.

**Or use the automated script:**
```bash
python scripts/migrate_mlflow_to_sqlite.py
```

---

## Benefits of SQLite Backend

✅ **Better Performance:** Faster query execution for experiments and models
✅ **ACID Transactions:** Proper database guarantees for concurrent access
✅ **Easier Backups:** Single file to backup/restore (`mlflow.db`)
✅ **Future-Proof:** Filesystem backend is deprecated (Feb 2026)
✅ **Same Features:** All MLflow features work identically

---

## Configuration Reference

### Main Pipeline Config

In `src/model_eval_suite/config/default_config.yaml`:

```yaml
base_config: &base_config
  mlflow_tracking_uri: "sqlite:///mlflow.db"  # Default
```

### Validation Config

In `config/xgb_validation.yaml` and other validation configs:

```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
```

### Config Schema

The `SuiteConfig` class now includes:

```python
class SuiteConfig(BaseModel):
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    # ... other fields
```

---

## Troubleshooting

### "Database is locked" error

If you see this error, it means another process is using the database:

```bash
# Check for running processes
ps aux | grep mlflow

# Kill any mlflow server processes
pkill -f "mlflow"
```

### Can't find my old experiments

Make sure you've migrated the data (see Option 3 above), or check if you're pointing to the right backend:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print(mlflow.search_experiments())
```

### Want to use a different database location

```yaml
mlflow_tracking_uri: "sqlite:////absolute/path/to/mlflow.db"
```

Or for PostgreSQL/MySQL (advanced):
```yaml
mlflow_tracking_uri: "postgresql://user:password@localhost/mlflow_db"
```

---

## Rollback Instructions

If you need to revert to filesystem backend temporarily:

1. **Restore your backup:**
   ```bash
   mv mlruns_old_backup mlruns
   ```

2. **Update your config:**
   ```yaml
   mlflow_tracking_uri: "file:./mlruns"
   ```

3. **Run your notebook/pipeline as usual**

---

## Questions?

- **Documentation:** [resource_hub/modeling_resources/MLFlow.md](resource_hub/modeling_resources/MLFlow.md)
- **Migration Script:** `scripts/migrate_mlflow_to_sqlite.py`
- **MLflow Docs:** https://mlflow.org/docs/latest/tracking.html#backend-stores

---

**Last Updated:** 2026-01-30
**Version:** v0.2.0
