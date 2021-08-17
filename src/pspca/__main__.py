"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Prjective Space PCA."""


if __name__ == "__main__":
    main(prog_name="pspca")  # pragma: no cover
