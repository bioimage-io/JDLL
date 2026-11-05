"""Bounded retries for transient package transport failures, not dependency errors."""

import subprocess
import time


def run_install(command, attempts=3):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    transient = ("operation timed out", "request timed out", "connection reset",
                 "connection aborted", "connection closed before message completed",
                 "temporary failure in name resolution", "502 bad gateway",
                 "503 service unavailable", "504 gateway timeout")
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        print(result.stdout, end="", flush=True)
        if result.returncode == 0:
            return
        if attempt == attempts or not any(message in result.stdout.lower() for message in transient):
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
        delay = 5 * attempt
        print(f"Transient package download failure; retrying in {delay}s ({attempt}/{attempts}).", flush=True)
        time.sleep(delay)
