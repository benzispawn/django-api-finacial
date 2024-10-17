#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from dotenv import load_dotenv


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api_financial.settings')
    load_dotenv()  # Load environment variables from .env
    # for key, value in os.environ.items():
    #     print(f"{key}: {value}")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    if len(sys.argv) > 1 and sys.argv[1] == 'runserver':
        # Get the port from environment or use default '8200'
        port = os.getenv('PORT', '8200')
        print(port)
        print(sys.argv)
        if len(sys.argv) == 2:  # If no port is provided, append the port
            sys.argv.append(port)
        elif len(sys.argv) == 3:  # If only the address is provided, append the port
            sys.argv[2] = f"127.0.0.1:{port}"

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
