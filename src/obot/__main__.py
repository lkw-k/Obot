"""Allow `python -m obot` as an alternative to the `obot` launcher."""

from obot.cli import app

if __name__ == "__main__":
    app(prog_name="obot")
