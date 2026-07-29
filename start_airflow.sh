#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$PROJECT_DIR/.venv_airflow/bin/activate"

export AIRFLOW_HOME="$PROJECT_DIR/airflow"

export PYTHONPATH="$PROJECT_DIR"

airflow standalone