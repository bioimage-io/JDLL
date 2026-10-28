"""Render the same classpath templates and format arguments used by Java."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
try:
    import tomllib
except ImportError:  # Python 3.10 development/test environments.
    import tomli as tomllib

TOMLS = Path(__file__).resolve().parents[2] / "src/main/resources/tomls"


def render(template: Path, name: str, cuda: str = "12.4", environment: str = "default") -> str:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("Use a nonempty alphanumeric workspace name")
    text = template.read_text(encoding="utf-8")
    args = [name]
    if template.name.endswith("-cuda.toml") and not template.name.endswith("-no-cuda.toml"):
        if cuda not in {"11.8", "12.1", "12.4"}:
            raise ValueError("CUDA must be 11.8, 12.1 or 12.4")
        args.extend([cuda.replace(".", ""), cuda.replace(".", "")])
    if template.name != "jdll-pixi-stardist.toml":
        if environment == "default":
            features = ["runtime", "pytorch", "core"]
        elif environment in {"macos-big-sur", "macos-catalina"}:
            if template.name != "jdll-pixi-mac-x86.toml":
                raise ValueError("Older host profiles are only available on Intel macOS")
            features = ["runtime", environment.replace("macos-", "pytorch-"), "core"]
        else:
            raise ValueError("Unknown JDLL environment: " + environment)
        args.append(json.dumps(features))
    text = text % tuple(args)
    parsed = tomllib.loads(text)
    if parsed["workspace"]["name"] != name:
        raise ValueError("Rendered workspace name is inconsistent")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, choices=sorted(p.name for p in TOMLS.glob("jdll-pixi-*.toml")))
    parser.add_argument("--name", default="jdll-ci")
    parser.add_argument("--cuda", default="12.4")
    parser.add_argument("--environment", default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(TOMLS / args.template, args.name, args.cuda, args.environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.template != "jdll-pixi-stardist.toml":
        source = TOMLS.parent / "python/opencv-python-compat/pyproject.toml"
        target = args.output.parent / "opencv-python-compat/pyproject.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    args.output.write_text(text, encoding="utf-8")
    print(f"Rendered {args.template} to {args.output}")


if __name__ == "__main__":
    main()
