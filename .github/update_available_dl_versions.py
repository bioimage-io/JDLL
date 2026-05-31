#!/usr/bin/env python3
"""Update JDLL engine jar URLs from Maven metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import os
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
    published_at: str
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


@dataclass(frozen=True)
class ResolvedJar:
    jar_version: str
    published_at: str


def timestamp_from_last_modified(url: str, timeout: int) -> str:
    if url.startswith("file://"):
        path = urllib.request.url2pathname(url[len("file://"):])
        return dt.datetime.fromtimestamp(os.path.getmtime(path), tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "jdll-engine-version-updater"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            last_modified = response.headers.get("Last-Modified")
    except urllib.error.HTTPError as exc:
        if exc.code != 405:
            raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
        request = urllib.request.Request(url, headers={"User-Agent": "jdll-engine-version-updater"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            last_modified = response.headers.get("Last-Modified")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc.reason}") from exc

    if not last_modified:
        raise RuntimeError(f"No Last-Modified header for {url}")
    parsed = email.utils.parsedate_to_datetime(last_modified)
    return parsed.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def declared_versions(metadata: ET.Element) -> list[str]:
    versions = [node.text.strip() for node in metadata.findall("versioning/versions/version") if node.text]
    if not versions:
        raise RuntimeError("Maven metadata does not declare any versions")
    return versions


def snapshot_jar(snapshot_metadata: ET.Element, artifact: str) -> ResolvedJar:
    for node in snapshot_metadata.findall("versioning/snapshotVersions/snapshotVersion"):
        extension = text(node, "extension")
        classifier = text(node, "classifier")
        value = text(node, "value")
        updated = text(node, "updated")
        if extension == "jar" and classifier is None and value:
            if not updated:
                raise RuntimeError(f"Snapshot jar metadata for {artifact} does not declare updated timestamp")
            return ResolvedJar(jar_version=value, published_at=updated)

    timestamp = text(snapshot_metadata, "versioning/snapshot/timestamp")
    build_number = text(snapshot_metadata, "versioning/snapshot/buildNumber")
    version = text(snapshot_metadata, "version")
    if timestamp and build_number and version and version.endswith("-SNAPSHOT"):
        return ResolvedJar(
            jar_version=f"{version[:-9]}-{timestamp}-{build_number}",
            published_at=timestamp.replace(".", ""),
        )

    raise RuntimeError(f"Could not determine latest snapshot jar version for {artifact}")


def release_jar(artifact_path: str, artifact: str, version: str, timeout: int) -> ResolvedJar:
    jar_url = f"{artifact_path}/{version}/{artifact}-{version}.jar"
    return ResolvedJar(jar_version=version, published_at=timestamp_from_last_modified(jar_url, timeout))


def latest_engine_version(repository: str, artifact: str, timeout: int) -> EngineVersion:
    artifact_path = repository_path(repository, artifact)
    metadata = fetch_xml(f"{artifact_path}/maven-metadata.xml", timeout)
    candidates: list[EngineVersion] = []

    for version in declared_versions(metadata):
        if version.endswith("-SNAPSHOT"):
            version_metadata = fetch_xml(f"{artifact_path}/{version}/maven-metadata.xml", timeout)
            resolved = snapshot_jar(version_metadata, artifact)
        else:
            resolved = release_jar(artifact_path, artifact, version, timeout)
        jar_url = f"{artifact_path}/{version}/{artifact}-{resolved.jar_version}.jar"
        candidates.append(
            EngineVersion(
                artifact=artifact,
                version=version,
                published_at=resolved.published_at,
                jar_url=jar_url,
            )
        )

    return max(candidates, key=lambda candidate: candidate.published_at)


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
