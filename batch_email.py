#!/usr/bin/env python3
"""
batch_email_final.py
-----------------------------------------------------
✅ Final stable version for Windows, macOS, and Linux.
✅ Auto UTF-8 handling (no more UnicodeEncodeError).
✅ Streams subprocess output safely.
✅ Works with your smart_email_finder.py v4+ structure.
"""

import os
import sys
import csv
import re
import time
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------
# 🔧 GLOBAL CONFIG
# ----------------------------------------------------------
SMART_SCRIPT = "smart_email_finder.py"
OUTPUT_FILE = "batch_results_final.csv"
MAX_WORKERS = 3
TIMEOUT = 90  # seconds per case

HEADERS = [
    "timestamp", "url", "person", "email_found",
    "profile", "status", "runtime_sec", "error"
]

# ----------------------------------------------------------
# 🧩 AUTO UTF-8 ENCODING FIX FOR WINDOWS
# ----------------------------------------------------------
if os.name == "nt":  # Windows only
    os.system("chcp 65001 >NUL")  # set console to UTF-8
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------
def run_case(url, person):
    """Run one case of smart_email_finder.py and extract email info."""
    start = time.time()
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": url.strip(),
        "person": person.strip(),
        "email_found": "",
        "profile": "",
        "status": "failed",
        "runtime_sec": 0,
        "error": ""
    }

    try:
        cmd = [sys.executable, SMART_SCRIPT, url.strip(), person.strip()]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )

        full_output = []
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            line = line.strip()
            print(line)  # stream output to console live
            full_output.append(line)

            # --- Parse direct emails ---
            if "@" in line and "." in line:
                emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", line)
                if emails:
                    result["email_found"] = emails[0]
                    result["status"] = "success"

            # --- Parse structured RESULT dictionary ---
            if "RESULT (direct)" in line:
                match = re.search(r"\{.*\}", line)
                if match:
                    try:
                        data = eval(match.group(0))
                        result["profile"] = data.get("profile", "")
                        emails = data.get("emails_found", [])
                        if emails:
                            result["email_found"] = emails[0]
                            result["status"] = "success"
                    except Exception as e:
                        result["error"] = f"Parse error: {e}"

        proc.wait(timeout=TIMEOUT)
        result["runtime_sec"] = round(time.time() - start, 2)

        if result["status"] == "failed" and not result["email_found"]:
            result["status"] = "no_email"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Timeout after {TIMEOUT}s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result

# ----------------------------------------------------------
def load_input_file(file_path):
    """Read input file lines (url, person)."""
    lines = [l.strip() for l in open(file_path, encoding="utf-8") if l.strip()]
    pairs = []
    for line in lines:
        if "," not in line:
            print(f"⚠️ Skipping invalid line: {line}")
            continue
        url, person = line.split(",", 1)
        pairs.append((url.strip(), person.strip()))
    return pairs

# ----------------------------------------------------------
def save_results(results):
    """Save results to CSV."""
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📁 Results saved to: {OUTPUT_FILE}")

# ----------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_email_final.py input.txt")
        sys.exit(1)

    infile = sys.argv[1]
    pairs = load_input_file(infile)
    print(f"🚀 Running batch for {len(pairs)} cases...\n")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_case, url, person): (url, person) for url, person in pairs}
        for i, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            results.append(res)
            email = res['email_found'] or 'None'
            print(f"[{i}/{len(pairs)}] ✅ {res['person']} @ {res['url']} → {res['status']} ({email})")

    print("\n==============================")
    print(f"✅ Completed {len(results)} total tests.")
    print("==============================")
    save_results(results)

# ----------------------------------------------------------
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
