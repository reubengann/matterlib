import argparse
import shutil
import sys
from pathlib import Path


def find_profile_default() -> Path:
    """
    Return the IPython profile_default directory cross-platform.
    Prefer IPython's own path logic if available.
    """
    try:
        from IPython.paths import get_ipython_dir  # type: ignore

        ipdir = Path(get_ipython_dir())
        return ipdir / "profile_default"
    except Exception:
        # Fallback: common default location
        return Path.home() / ".ipython" / "profile_default"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install a matplotlib/Jupyter setup script into IPython startup."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("jupyterlab_setup.py"),
        help="Path to the setup .py file to install (default: ./jupyterlab_setup.py)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="00-jupyterlab-setup.py",
        help="Filename to use in startup/ (default: 00-jupyterlab-setup.py)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help='IPython profile name (default: "default" -> profile_default)',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination file if present.",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        return 2

    ipdir = find_profile_default().parent  # .../.ipython
    profile_dir = ipdir / f"profile_{args.profile}"
    startup_dir = profile_dir / "startup"
    dst = startup_dir / args.name

    print(f"Source:      {src}")
    print(f"IPython dir:  {ipdir}")
    print(f"Profile dir:  {profile_dir}")
    print(f"Startup dir:  {startup_dir}")
    print(f"Destination: {dst}")

    if args.dry_run:
        print("Dry run: no files copied.")
        return 0

    startup_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not args.force:
        print(
            f"ERROR: destination already exists: {dst}\n" f"Use --force to overwrite.",
            file=sys.stderr,
        )
        return 3

    shutil.copy2(src, dst)
    print("Installed. Restart JupyterLab completely for it to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
