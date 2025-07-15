# Justfile for OrderFlow Pro
set dotenv-load

# Variables
python := "poetry run python"
pytest := "poetry run pytest"

# Default recipe to display help
default:
    @just --list

# Install dependencies using Poetry
install:
    poetry install --with dev

# Format code using ruff and isort
format:
    poetry run ruff check . --select I --fix # sort imports
    poetry run ruff format .

# Run linter for python
lint:
    poetry run ruff check src tests

# Run tests
test:
    {{ pytest }} -v

# Run unit tests
u_test *arguments:
    poetry run pytest tests/unit {{arguments}}

# Run integration tests
i_test *arguments:
    poetry run pytest tests/integration {{arguments}}

# Run all checks
all: format lint u_test i_test

# Setup development environment
setup-dev:
    poetry install --with dev
    cp .env.example .env
    @echo "✅ Development environment setup complete!"
    @echo "📝 Don't forget to update .env with your API keys"

# Run the application (dev)
run:
    {{ python }} -m orderflow_pro.main

# Run with debug logging
run-debug:
    LOGLEVEL=DEBUG {{ python }} -m orderflow_pro.main

# Run with auto-reload for development
dev:
    poetry run watchfiles '{{ python }} -m orderflow_pro.main' src/


# Clean up caches and build artifacts
clean:
    find . -type d -name "__pycache__" -delete
    find . -type f -name "*.pyc" -delete
    rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/

# Clean up virtual environment and reinstall
clean-venv:
    rm -rf .venv
    rm -f poetry.lock
    poetry install --with dev

# Monitor application logs
logs:
    tail -f logs/orderflow_pro.log

# Build package
build:
    poetry build

# Setup pre-commit hooks
setup-hooks:
    poetry run pre-commit install
    @echo "✅ Pre-commit hooks installed!"

# Run pre-commit on all files
pre-commit-all:
    poetry run pre-commit run --all-files

# Docker commands
docker-build:
    docker build -t orderflow-pro .

docker-run:
    docker-compose up -d

docker-stop:
    docker-compose down

docker-logs:
    docker-compose logs -f orderflow-pro

docker-shell:
    docker-compose exec orderflow-pro /bin/bash