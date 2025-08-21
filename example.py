import os
from config.settings import settings, get_settings

def print_settings_for_env(env_name: str):
    """Sets APP_ENV, clears cache, and prints the reloaded settings."""
    # Set the environment variable
    os.environ["APP_ENV"] = env_name
    # Clear the cache of the loader function to force a reload
    get_settings.cache_clear()

    print(f"\n--- Loading settings for '{env_name}' environment ---")

    try:
        # Call the loader function again to get the reloaded settings
        current_settings = get_settings()
        print("Settings loaded successfully!")
        print(f"API URL: {current_settings.api.url}")
        print(f"API Key: {current_settings.api.key}")
        print(f"Model Name: {current_settings.model.name}")
        print(f"Temperature: {current_settings.model.temperature}")
        print(f"Debug Mode: {current_settings.system.debug}")
        print("-" * 30)
    except FileNotFoundError as e:
        print(f"Error: Could not load settings. {e}")
        print("-" * 30)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("-" * 30)


if __name__ == "__main__":
    # The `settings` object is loaded when config.settings is first imported.
    # By default, this will be the 'development' environment.
    print("--- Initial settings (should be 'development') ---")
    print(f"API URL: {settings.api.url}")
    print(f"Debug Mode: {settings.system.debug}")
    print("-" * 30)

    # Now, demonstrate reloading for 'production'
    print_settings_for_env("production")

    # Demonstrate reloading back to 'development'
    print_settings_for_env("development")

    # Demonstrate handling a missing environment file
    print_settings_for_env("staging")
