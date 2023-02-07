import json
from pathlib import Path
import platform
import sys
import traceback
import urllib.request

# HACK: Disable SSL certification validation for urllib.
# Because otherwise we see for all attempts ;-) :
# urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1108)>
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

engine_filter = sys.argv[1] if len(sys.argv) > 1 else ''
version_filter = sys.argv[2] if len(sys.argv) > 2 else ''

with open("src/main/resources/availableDLVersions.json") as f:
    data = json.load(f)

# TODO: Make sure this maps to "windows" for Windows...
my_os = platform.system().lower()
if my_os == "darwin":
    my_os = "macosx"
my_os = f"{my_os}-{platform.machine()}"

for entry in data["versions"]:
    engine = entry["engine"]
    version = entry["version"]
    pythonVersion = entry["pythonVersion"]

    if (
        engine_filter.lower() not in engine.lower() or
        version_filter.lower() not in version.lower()
    ):
        continue

    os = entry["os"]
    if os != my_os:
        continue

    gpu = "-gpu" if entry["gpu"] else ""
    cpu = "-cpu" if entry["cpu"] else ""
    rosetta = entry["rosetta"]
    jars = entry["jars"]

    # From the README:
    # <DL_framework_name>.<python_version>.<java_api_version>.<os>.<architecture>.<cpu_if_it_runs_in_cpu>.<gpu_if_it_runs_in_gpu>.
    # But it's actually dashes, not dots.
    folder_name = f"{engine}-{pythonVersion}-{version}-{os}{cpu}{gpu}"

    print(folder_name)

    # download each JAR into the folder
    engine_dir = Path(".") / "engines" / folder_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    for jar_url in jars:
        jar_name = Path(jar_url).name
        jar_path = engine_dir / jar_name
        if jar_path.exists():
            continue
        print(f"{engine_dir}: downloading {jar_url}")
        try:
            urllib.request.urlretrieve(jar_url, jar_path)
        except:
            print(f"[ERROR] Failed to download {jar_url}")
            traceback.print_exc()
