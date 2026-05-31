#!/usr/bin/env python3
"""Update JDLL engine jar URLs from Maven metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPOSITORY = "https://maven.scijava.org/content/groups/public"
DEFAULT_JSON = Path("src/main/resources/availableDLVersions.json")

ENGINE_ARTIFACTS = (
    "dl-modelrunner-tensorflow-1",
    "dl-modelrunner-tensorflow-2a",
    "dl-modelrunner-tensorflow-2b",
    "dl-modelrunner-tensorflow-2c",
    "dl-modelrunner-pytorch-javacpp",
    "dl-modelrunner-pytorch",
    "dl-modelrunner-onnx",
)


@dataclass(frozen=True)
class EngineVersion:
    artifact: str
    version: str
    jar_url: str


def fetch_xml(url: str, timeout: int) -> ET.Element:
    request = urllib.request.Request(url, headers={"User-Agent": "jdll-engine-version-updater"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return ET.fromstring(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc.reason}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid Maven metadata XML at {url}") from exc


def text(element: ET.Element, path: str) -> str | None:
    child = element.find(path)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def repository_path(repository: str, artifact: str) -> str:
    return f"{repository.rstrip('/')}/io/bioimage/{artifact}"


def latest_declared_version(metadata: ET.Element) -> str:
    latest = text(metadata, "versioning/latest")
    if latest:
        return latest
    versions = [node.text.strip() for node in metadata.findall("versioning/versions/version") if node.text]
    if not versions:
        raise RuntimeError("Maven metadata does not declare any versions")
    return versions[-1]


def snapshot_jar_version(snapshot_metadata: ET.Element, artifact: str) -> str:
    for node in snapshot_metadata.findall("versioning/snapshotVersions/snapshotVersion"):
        extension = text(node, "extension")
        classifier = text(node, "classifier")
        value = text(node, "value")
        if extension == "jar" and classifier is None and value:
            return value

    timestamp = text(snapshot_metadata, "versioning/snapshot/timestamp")
    build_number = text(snapshot_metadata, "versioning/snapshot/buildNumber")
    version = text(snapshot_metadata, "version")
    if timestamp and build_number and version and version.endswith("-SNAPSHOT"):
        return f"{version[:-9]}-{timestamp}-{build_number}"

    raise RuntimeError(f"Could not determine latest snapshot jar version for {artifact}")


def latest_engine_version(repository: str, artifact: str, timeout: int) -> EngineVersion:
    artifact_path = repository_path(repository, artifact)
    metadata = fetch_xml(f"{artifact_path}/maven-metadata.xml", timeout)
    version = latest_declared_version(metadata)

    jar_version = version
    if version.endswith("-SNAPSHOT"):
        snapshot_metadata = fetch_xml(f"{artifact_path}/{version}/maven-metadata.xml", timeout)
        jar_version = snapshot_jar_version(snapshot_metadata, artifact)

    jar_url = f"{artifact_path}/{version}/{artifact}-{jar_version}.jar"
    return EngineVersion(artifact=artifact, version=version, jar_url=jar_url)


def replace_engine_urls(json_path: Path, versions: dict[str, EngineVersion]) -> bool:
    original = json_path.read_text(encoding="utf-8")
    json.loads(original)

    updated = original
    changed = False
    for artifact, version in versions.items():
        pattern = re.compile(
            r"https://[^\"\s]+/io/bioimage/"
            + re.escape(artifact)
            + r"/[^/\"\s]+/"
            + re.escape(artifact)
            + r"-[^/\"\s]+\.jar"
        )
        matches = sorted(set(pattern.findall(updated)))
        if not matches:
            raise RuntimeError(f"No jar URL found in {json_path} for {artifact}")
        for current_url in matches:
            if current_url != version.jar_url:
                print(f"{artifact}: {current_url} -> {version.jar_url}")
                updated = updated.replace(current_url, version.jar_url)
                changed = True
            else:
                print(f"{artifact}: already at {version.jar_url}")

    if changed:
        json.loads(updated)
        json_path.write_text(updated, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    versions: dict[str, EngineVersion] = {}
    for artifact in ENGINE_ARTIFACTS:
        versions[artifact] = latest_engine_version(args.repository, artifact, args.timeout)

    changed = replace_engine_urls(args.json, versions)
    print("Updated availableDLVersions.json" if changed else "No engine jar URL updates needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
