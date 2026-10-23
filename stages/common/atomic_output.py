"""Publish complete stage artifacts without changing their file permissions."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import stat
import tempfile


@contextmanager
def atomic_output_path(destination: Path) -> Iterator[Path]:
    """Replace an output only after its writer and validation finish successfully.

    Parameters
    ----------
    destination : Path
        Final output path. Missing parent directories are created.

    Yields
    ------
    Path
        Temporary sibling path to create, write, validate, and close before leaving
        the context. Normal file creation applies umask to new outputs. Replacements
        retain the existing destination's permission bits.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        yield temporary
        if destination.exists():
            temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
        temporary.replace(destination)
