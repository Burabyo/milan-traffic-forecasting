"""Download the Milan CDR files from Dataverse, handling the dataset guestbook.

Two obstacles make the obvious approaches fail, and both are worth knowing:

1. **The browser ZIP route cannot work.** The dataset is 19.4 GB and Harvard
   Dataverse caps server-side ZIP bundles at 15 GB. Splitting into batches is
   possible but fragile: the bundle is streamed while being zipped, so any
   interruption loses the whole transfer with nothing resumable.

2. **The dataset is guestbook-protected.** Guestbook 96 ("Privacy risk
   assessment") requires an email address before any file is released. A plain
   ``GET /api/access/datafile/<id>`` therefore returns 400. The documented route
   is to POST the guestbook response to the same endpoint, which returns a
   short-lived **signed URL**, and to download from that.

Because signed URLs expire, this script requests one immediately before each
transfer rather than collecting all 62 in advance, and requests a fresh one on
every retry. Downloads are resumable via HTTP Range, so an interrupted run
continues from the exact byte it stopped at.

Setup:
    1. Create a free account at https://dataverse.harvard.edu
    2. Your name -> API Token -> Create Token
    3. export DV_TOKEN='your-token'

Usage:
    python -m scripts.download_dataset --out-dir ~/Documents/traffic-forecasting/Dataverse

Safe to re-run: completed files are skipped, partial files resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/EGZHFV"
METADATA_URL = f"{SERVER}/api/datasets/export?exporter=dataverse_json&persistentId={DOI}"
ACCESS_URL = SERVER + "/api/access/datafile/{fid}"

NAME_PREFIX = "sms-call-internet-mi-"
EXPECTED_COUNT = 62

# Dataverse's front end rejects the default ``Python-urllib/3.x`` user-agent
# with a 403, so the client identifies itself in a form the server accepts.
USER_AGENT = "milan-traffic-forecasting/2.0 (academic research) Mozilla/5.0 compatible"


def _request(url: str, *, data: bytes | None = None, method: str | None = None,
             token: str | None = None, extra: dict | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", USER_AGENT)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Dataverse-key", token)
    for key, value in (extra or {}).items():
        req.add_header(key, value)
    return req


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------
def fetch_metadata(url: str = METADATA_URL) -> list[dict]:
    """Return [{id, name, size}] for every daily traffic file in the dataset."""
    print("Fetching dataset file list ...")
    with urllib.request.urlopen(_request(url), timeout=120) as resp:
        meta = json.load(resp)

    files = []
    for entry in meta.get("datasetVersion", {}).get("files", []):
        df = entry.get("dataFile", {})
        name = df.get("filename") or entry.get("label") or ""
        if not name.startswith(NAME_PREFIX):
            continue
        files.append({"id": df.get("id"), "name": name,
                      "size": int(df.get("filesize") or 0),
                      "md5": df.get("md5")})
    files.sort(key=lambda f: f["name"])
    return files


def get_signed_url(file_id: int, token: str, email: str | None) -> str:
    """Submit the guestbook response and return a signed download URL.

    The response is short-lived (the URL carries an ``until`` timestamp), so it
    is requested immediately before the transfer it is used for.
    """
    payload: dict = {"guestbookResponse": {}}
    if email:
        payload["guestbookResponse"]["email"] = email
    req = _request(ACCESS_URL.format(fid=file_id),
                   data=json.dumps(payload).encode(), method="POST", token=token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.load(resp)
    url = body.get("data", {}).get("signedUrl")
    if not url:
        raise IOError(f"no signedUrl in response: {str(body)[:200]}")
    return url


def download_one(info: dict, out_dir: Path, token: str, email: str | None,
                 retries: int = 5) -> bool:
    dest = out_dir / info["name"]
    expected = info["size"]

    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"  {info['name']}: already complete, skipping")
        return True

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have > expected:
            print(f"  {info['name']}: local file oversized, restarting")
            dest.unlink()
            have = 0
        if expected and have == expected:
            return True

        try:
            # A fresh signed URL every attempt: the previous one may have expired
            # mid-transfer, which is precisely when a retry happens.
            url = get_signed_url(info["id"], token, email)

            extra = {"Range": f"bytes={have}-"} if have else {}
            if have:
                print(f"  {info['name']}: resuming at {human(have)}")
            req = _request(url, extra=extra)
            mode = "ab" if have else "wb"

            with urllib.request.urlopen(req, timeout=300) as resp:
                if have and resp.status != 206:
                    # Server ignored the Range header. Restart cleanly rather
                    # than splicing two partial bodies into a corrupt file.
                    print(f"  {info['name']}: resume refused, restarting")
                    dest.unlink(missing_ok=True)
                    have, mode = 0, "wb"

                written, t0, last = have, time.time(), 0.0
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if expected and now - last > 0.5:
                            last = now
                            rate = (written - have) / max(now - t0, 1e-6)
                            eta = (expected - written) / rate if rate > 0 else 0
                            print(f"\r  {info['name']}: {written / expected * 100:5.1f}% "
                                  f"({human(written)}/{human(expected)}) "
                                  f"{human(rate)}/s ETA {eta / 60:.0f}m   ",
                                  end="", flush=True)
                print()

            if expected and dest.stat().st_size != expected:
                raise IOError(f"size mismatch: {dest.stat().st_size} != {expected}")
            return True

        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            wait = min(2 ** attempt, 60)
            print(f"\n  {info['name']}: attempt {attempt}/{retries} failed ({exc}); "
                  f"retrying in {wait}s")
            time.sleep(wait)

    print(f"  {info['name']}: FAILED after {retries} attempts")
    return False


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the Milan CDR dataset.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--email", default=None,
                    help="guestbook email; defaults to your Dataverse account")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args(argv)

    token = os.environ.get("DV_TOKEN", "").strip()
    if not token:
        print("No API token found. The dataset requires a guestbook response, "
              "which needs an account:")
        print("  1. Sign up at https://dataverse.harvard.edu")
        print("  2. Your name -> API Token -> Create Token")
        print("  3. export DV_TOKEN='your-token'")
        return 1

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = fetch_metadata()
    except urllib.error.HTTPError as exc:
        print(f"Could not fetch the file list: HTTP {exc.code} {exc.reason}")
        return 1
    except Exception as exc:
        print(f"Could not fetch the file list: {exc}")
        return 1

    total = sum(f["size"] for f in files)
    print(f"Found {len(files)} files, {human(total)} total")
    if len(files) != EXPECTED_COUNT:
        print(f"  warning: expected {EXPECTED_COUNT} files")

    failed = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]")
        if not download_one(f, out_dir, token, args.email, args.retries):
            failed.append(f["name"])

    have = sorted(out_dir.glob(f"{NAME_PREFIX}*.txt"))
    print("\n" + "=" * 60)
    print(f"Files in {out_dir}: {len(have)}")
    print(f"Total size: {human(sum(p.stat().st_size for p in have))}")

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for name in failed:
            print(f"  {name}")
        print("Re-run the same command; completed files skip and partial ones resume.")
        return 1

    if len(have) != EXPECTED_COUNT:
        print(f"\nWARNING: {len(have)} files present, expected {EXPECTED_COUNT}.")
        return 1

    print("\nAll files downloaded. Next:")
    print(f"  python -m src.ingest --raw-dir {out_dir} --benchmark --chunksize 4000000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
