#!/usr/bin/env sh
# venvify.sh — recreate and activate a Python venv from pyproject .[dev]
# Works in bash and zsh (and other POSIX shells).

set -e

# --- Config ---------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"     # change by exporting VENV_DIR=... before running
PYTHON_BIN="${PYTHON:-python3}"   # override with PYTHON=python3.12, etc.

# --- Helpers --------------------------------------------------------------
say() { printf "\033[1;34m[setup_venv]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[setup_venv]\033[0m %s\n" "$*" >&2; }

is_command() { command -v "$1" >/dev/null 2>&1; }

is_sourced() {
  # Detect if this script is sourced (works for bash & zsh; harmless elsewhere)
  # 1) In bash, 'return' is only valid when sourced.
  (return 0 2>/dev/null) && return 0
  # 2) In zsh, ZSH_EVAL_CONTEXT contains ":file" when sourced.
  [ -n "${ZSH_EVAL_CONTEXT:-}" ] && printf %s "$ZSH_EVAL_CONTEXT" | grep -q ":file" && return 0
  return 1
}

# --- Preconditions --------------------------------------------------------
[ -f "pyproject.toml" ] || { err "pyproject.toml not found in current directory."; exit 1; }

if ! is_command "$PYTHON_BIN"; then
  # Fallback to 'python' if 'python3' not present
  if is_command python; then
    PYTHON_BIN=python
  else
    err "Python not found. Install python3 or set PYTHON=<path>."
    exit 1
  fi
fi

# --- Remove existing venv -------------------------------------------------
if [ -d "$VENV_DIR" ]; then
  say "Removing existing virtual environment: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

# --- Create venv ----------------------------------------------------------
say "Creating virtual environment in $VENV_DIR using $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# --- Activate venv in this process ---------------------------------------
# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

# --- Upgrade pip/setuptools/wheel & install deps --------------------------
say "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

say "Installing project with dev extras: -e '.[dev]'"
# If your project doesn't define a [project.optional-dependencies].dev,
# this will error out. That’s intentional so you notice.
pip install -e ".[dev]"

say "✅ Virtual environment ready at $VENV_DIR"

# --- Auto-launch behavior -------------------------------------------------
# If the script is *executed*, we want to drop the user into an interactive shell
# that inherits the activated environment. If it's *sourced*, the current shell
# is already activated so we just return.
if is_sourced; then
  say "Activated in the current shell. (Use 'deactivate' to exit.)"
  return 0 2>/dev/null || exit 0
else
  # Start a fresh interactive shell that keeps this environment.
  # We use '-c "source ...; exec $SHELL -i"' to ensure the final shell is interactive
  # and inherits the activated environment variables (PATH, VIRTUAL_ENV, etc.).
  say "Launching a new interactive shell with the venv activated…"
  # Default to user's shell, fallback to bash.
  TARGET_SHELL="${SHELL:-/bin/bash}"
  exec "$TARGET_SHELL" -i -c ". \"$VENV_DIR/bin/activate\"; exec \"$TARGET_SHELL\" -i"
fi
