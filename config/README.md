# Configuration Management

This directory contains the configuration management system for the application.

## How it works

The configuration is managed by the `Settings` class in `settings.py`, which uses `pydantic` for validation and type hinting.

The system loads configuration from environment-specific `.env` files. The environment is determined by the `APP_ENV` environment variable. For example, if `APP_ENV` is set to `production`, the system will load settings from `production.env`. If `APP_ENV` is not set, it defaults to `development`.

## Usage

To use the configuration in your application, import the `settings` object from `config.settings`:

```python
from config.settings import settings

# Access your settings
api_url = settings.api.url
api_key = settings.api.key
```

## Adding new settings

To add a new setting, open `config/settings.py` and add the new field to the appropriate settings class (e.g., `ApiSettings`, `ModelParameters`, or `SystemSettings`).

Then, add the corresponding key-value pair to the `development.env` and `production.env` files.

## Environments

- `development.env`: Settings for the development environment.
- `production.env`: Settings for the production environment.

You can create new environment files (e.g., `staging.env`) and set the `APP_ENV` environment variable accordingly to use them.
