import sys
import os
import inspect


def generate_snapshot():
    try:
        import zero_torch
    except ImportError:
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        # Also try to add compiler path if developing locally
        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "ml-switcheroo-compiler",
                    "src",
                )
            ),
        )
        import zero_torch

    apis = []
    for name, obj in inspect.getmembers(zero_torch):
        if not name.startswith("_") and (
            inspect.isfunction(obj) or inspect.isclass(obj)
        ):
            try:
                sig = inspect.signature(obj)
                apis.append(f"{name}{sig}")
            except ValueError:
                apis.append(f"{name}(...)")
    return "\n".join(sorted(apis)) + "\n"


def main():
    api_snapshot = generate_snapshot()
    snapshot_file = "ml_snapshots.txt"

    if not os.path.exists(snapshot_file):
        with open(snapshot_file, "w") as f:
            f.write(api_snapshot)
        print(f"Generated {snapshot_file}")
        sys.exit(0)

    with open(snapshot_file, "r") as f:
        expected = f.read()

    if api_snapshot != expected:
        print("API mismatch against snapshot!")
        if os.environ.get("UPDATE_SNAPSHOTS") == "1":
            with open(snapshot_file, "w") as f:
                f.write(api_snapshot)
            print(f"Updated {snapshot_file}")
            sys.exit(0)
        else:
            print(
                "Run with UPDATE_SNAPSHOTS=1 python scripts/check_api_snapshots.py to update."
            )
            sys.exit(1)

    print("API matches snapshot.")
    sys.exit(0)


if __name__ == "__main__":
    main()
