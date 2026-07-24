#!/usr/bin/env python3
"""Rent the cheapest verified H200 SXM on vast.ai, run the HBM3e power
measurement, fetch the results, and always tear the instance down.

The whole thing is non-interactive. The only required input is a vast.ai API
key, passed with --api-key or the VAST_API_KEY environment variable. There is
no web console and no prompting.

    python3 scripts/h200/vast_run.py --dry-run   # search + select only, rents nothing
    python3 scripts/h200/vast_run.py             # full run: rent -> measure -> destroy
    python3 scripts/h200/vast_run.py --keep      # leave the instance running afterwards

What runs on the GPU is sources/h200/quick_measure_h200.py with its defaults:
a ~30-40 s idle-baseline + random-read measurement that writes quick_summary.csv
(idle mem W, random-read mem W, active = read - idle, achieved GB/s).

NOTE ON ENDPOINTS: the calls below target vast.ai's public API v0
(https://console.vast.ai/api/v0). The paths and the /bundles query grammar are
the stable documented surface, but if vast changes the API this is the one place
to adjust -- every request goes through the small VastAPI class.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# vast.ai's API is mid-migration: most calls still live under v0, but the
# instance *list* moved to v1 (v0 /instances/ now returns 410 deprecated).
# Each API method below picks its base explicitly.
API_V0 = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"

# Compile image: must ship nvcc (a -devel tag) and support sm_90 (Hopper/H200).
# The base image is minimal, so python3 is installed on the instance at runtime.
DEFAULT_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

# The four files quick_measure_h200.py needs to run standalone on the GPU.
# This script lives at scripts/h200/, so the repo root is two levels up and the
# sources are at <repo>/sources/h200/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SOURCE_DIR = os.path.join(REPO_ROOT, "sources", "h200")
UPLOAD_FILES = [
    "hbm_bw_3.cu",
    "quick_measure_h200.py",
    "log_power.py",
    "summarize_tuning.py",
]

REMOTE_DIR = "/root/h200"          # where the sources land on the instance
REMOTE_OUT = "/root/h200/out"      # quick_measure_h200.py --outdir target

# Full verbose artifacts go here (raw per-sample CSVs, stdout logs); the single
# summary CSV is also published to the canonical dataset path below.
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "temp", "h200_results")
SUMMARY_DEST = os.path.join(REPO_ROOT, "data", "new", "h200", "hbm3e_results.csv")

# Offer selection -- mirrors the vast.ai web query you shared:
#   gpuModelNames=h200Sxm, DiskSizeMin=32, DurationMin=604800,
#   ReliabilityMin=0.9, versionCudaMin=12.1, verified, available, on-demand.
GPU_NAME = "H200"                  # SXM; "H200 NVL" is a different gpu_name
MIN_DISK_GB = 32
MIN_CUDA = 12.1
MIN_RELIABILITY = 0.9
MIN_DURATION_S = 604800            # host must offer >= 7 days
MAX_PRICE_DEFAULT = 5.0            # $/hr guard; refuse anything pricier

# Timeouts and polling cadence (seconds).
PROVISION_TIMEOUT = 900            # image pull + boot to actual_status == running
SSH_READY_TIMEOUT = 300            # sshd accepting connections after "running"
MEASURE_TIMEOUT = 900             # remote measurement wall-clock cap
POLL_INTERVAL = 10                 # how often to re-check status while waiting

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
]


# --------------------------------------------------------------------------- #
# vast.ai REST API                                                            #
# --------------------------------------------------------------------------- #

class VastAPI:
    """Thin wrapper over the vast.ai API. One method per call we make."""

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer " + api_key})

    def _request(self, method, path, base=API_V0, **kwargs):
        resp = self.session.request(method, base + path, timeout=30, **kwargs)
        if not resp.ok:
            raise RuntimeError(
                "vast API {} {} -> {}: {}".format(
                    method, path, resp.status_code, resp.text[:400]))
        return resp.json() if resp.content else {}

    def search_offers(self, query):
        """Return the list of matching offers (cheapest first per the query)."""
        result = self._request("GET", "/bundles/", params={"q": json.dumps(query)})
        return result.get("offers", [])

    def create_instance(self, offer_id, image, disk_gb):
        """Rent an offer. Returns the new instance (contract) id."""
        body = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "runtype": "ssh",
            "label": "hbm3e-power",
            "target_state": "running",
        }
        result = self._request("PUT", "/asks/{}/".format(offer_id), json=body)
        if not result.get("success"):
            raise RuntimeError("instance creation failed: {}".format(result))
        return result["new_contract"]

    def get_instance(self, instance_id):
        """Return the instance dict, or None if it is not listed (yet).

        Uses the v1 list endpoint; v0 /instances/ is deprecated (410).
        """
        result = self._request("GET", "/instances/", base=API_V1)
        for inst in result.get("instances", []):
            if inst.get("id") == instance_id:
                return inst
        return None

    def attach_ssh_key(self, instance_id, public_key):
        """Authorize our ephemeral key on this specific instance."""
        self._request("POST", "/instances/{}/ssh/".format(instance_id),
                      json={"ssh_key": public_key})

    def destroy_instance(self, instance_id):
        self._request("DELETE", "/instances/{}/".format(instance_id))


def build_query(max_price):
    """Translate the web-console filters into a /bundles query dict."""
    return {
        "verified":      {"eq": True},
        "rentable":      {"eq": True},
        "rented":        {"eq": False},
        "num_gpus":      {"eq": 1},
        "gpu_name":      {"eq": GPU_NAME},
        "disk_space":    {"gte": MIN_DISK_GB},
        "cuda_max_good": {"gte": MIN_CUDA},
        "reliability2":  {"gte": MIN_RELIABILITY},
        "duration":      {"gte": MIN_DURATION_S},
        "dph_total":     {"lte": max_price},
        "type":          "on-demand",
        "order":         [["dph_total", "asc"]],
        "limit":         64,
    }


def pick_cheapest(offers, max_price):
    """Choose the lowest $/hr offer, guarding against price and gpu-name drift."""
    candidates = [
        o for o in offers
        if o.get("gpu_name") == GPU_NAME and o.get("dph_total", 1e9) <= max_price
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: o["dph_total"])


def describe_offer(offer):
    return (
        "offer {id}: {ngpu}x {name}  ${price:.3f}/hr  "
        "cuda_max_good={cuda}  reliability={rel:.3f}  disk={disk:.0f}GB  {geo}"
        .format(
            id=offer.get("id"),
            ngpu=offer.get("num_gpus"),
            name=offer.get("gpu_name"),
            price=offer.get("dph_total", 0.0),
            cuda=offer.get("cuda_max_good"),
            rel=offer.get("reliability2", 0.0),
            disk=offer.get("disk_space", 0.0),
            geo=offer.get("geolocation", "?"),
        )
    )


# --------------------------------------------------------------------------- #
# SSH helpers                                                                 #
# --------------------------------------------------------------------------- #

def generate_ssh_key(directory):
    """Create an ephemeral ed25519 keypair. Returns (private_path, public_key)."""
    key_path = os.path.join(directory, "id_ed25519")
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", key_path],
        check=True,
    )
    with open(key_path + ".pub") as f:
        public_key = f.read().strip()
    return key_path, public_key


def ssh_base(key_path, host, port):
    return ["ssh", "-i", key_path, "-p", str(port)] + SSH_OPTS + ["root@" + host]


def ssh_run(key_path, host, port, command, timeout=MEASURE_TIMEOUT, stream=False):
    """Run a command on the instance. Streams output when stream=True."""
    argv = ssh_base(key_path, host, port) + [command]
    if stream:
        return subprocess.run(argv, timeout=timeout).returncode == 0
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stdout, result.stderr


def scp_up(key_path, host, port, local_paths, remote_dir):
    argv = ["scp", "-i", key_path, "-P", str(port)] + SSH_OPTS \
        + list(local_paths) + ["root@{}:{}".format(host, remote_dir)]
    subprocess.run(argv, check=True)


def scp_down(key_path, host, port, remote_path, local_dir):
    argv = ["scp", "-r", "-i", key_path, "-P", str(port)] + SSH_OPTS \
        + ["root@{}:{}".format(host, remote_path), local_dir]
    subprocess.run(argv, check=True)


# --------------------------------------------------------------------------- #
# Orchestration steps                                                         #
# --------------------------------------------------------------------------- #

def wait_until_running(api, instance_id):
    """Poll every POLL_INTERVAL s until actual_status == 'running'.

    Prints a timestamped status line each poll so provisioning progress is
    visible live (the loading -> running transition can take a few minutes).
    """
    start = time.time()
    while time.time() - start < PROVISION_TIMEOUT:
        elapsed = int(time.time() - start)
        inst = api.get_instance(instance_id)
        status = (inst or {}).get("actual_status") or "pending"
        message = ((inst or {}).get("status_msg", "") or "").strip()[:80]
        print("  [{:>3}s] status: {}  {}".format(elapsed, status, message))
        if status == "running":
            return inst
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("instance {} not running after {}s".format(
        instance_id, PROVISION_TIMEOUT))


def wait_for_ssh(key_path, host, port):
    """Poll every POLL_INTERVAL s until sshd answers, reporting progress."""
    start = time.time()
    while time.time() - start < SSH_READY_TIMEOUT:
        ok, _, _ = ssh_run(key_path, host, port, "true", timeout=20)
        if ok:
            print("  [{:>3}s] ssh ready".format(int(time.time() - start)))
            return
        print("  [{:>3}s] waiting for sshd...".format(int(time.time() - start)))
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("ssh to {}:{} not ready after {}s".format(
        host, port, SSH_READY_TIMEOUT))


def run_measurement(key_path, host, port, results_dir):
    """Install python3, upload sources, run quick_measure, fetch results back."""
    print(">> preparing instance (python3) and staging sources")
    ok = ssh_run(
        key_path, host, port,
        "export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3 && "
        "mkdir -p {}".format(REMOTE_DIR),
        timeout=300, stream=True,
    )
    if not ok:
        raise RuntimeError("remote setup (apt-get / mkdir) failed")

    local_files = [os.path.join(SOURCE_DIR, name) for name in UPLOAD_FILES]
    scp_up(key_path, host, port, local_files, REMOTE_DIR)

    print(">> running quick_measure_h200.py (nvcc build + idle + random-read)")
    ok = ssh_run(
        key_path, host, port,
        "cd {dir} && python3 quick_measure_h200.py --outdir {out}".format(
            dir=REMOTE_DIR, out=REMOTE_OUT),
        timeout=MEASURE_TIMEOUT, stream=True,
    )
    if not ok:
        raise RuntimeError("remote measurement failed")

    print(">> fetching results to {}".format(results_dir))
    os.makedirs(results_dir, exist_ok=True)
    scp_down(key_path, host, port, REMOTE_OUT, results_dir)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    # Line-buffer stdout so progress (status polls, step markers) shows up live
    # even when the output is redirected to a file or a background task log.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python < 3.7; prints just buffer as before

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", default=os.environ.get("VAST_API_KEY"),
                        help="vast.ai API key (or set VAST_API_KEY)")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="CUDA -devel image with nvcc (default: %(default)s)")
    parser.add_argument("--disk", type=int, default=MIN_DISK_GB,
                        help="instance disk in GB (default: %(default)s)")
    parser.add_argument("--max-price", type=float, default=MAX_PRICE_DEFAULT,
                        help="refuse offers above this $/hr (default: %(default)s)")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="local directory for fetched (verbose) results")
    parser.add_argument("--dry-run", action="store_true",
                        help="search and select only; rent nothing")
    parser.add_argument("--keep", action="store_true",
                        help="do not destroy the instance when finished")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("ERROR: no API key. Pass --api-key or set VAST_API_KEY.")

    api = VastAPI(args.api_key)

    # 1. Search + select (free, read-only) -------------------------------------
    print(">> searching for verified on-demand {} offers".format(GPU_NAME))
    offers = api.search_offers(build_query(args.max_price))
    offer = pick_cheapest(offers, args.max_price)
    if offer is None:
        sys.exit("No {} offer matched (<= ${:.2f}/hr, reliability >= {}, "
                 "cuda >= {}).".format(GPU_NAME, args.max_price,
                                       MIN_RELIABILITY, MIN_CUDA))
    print("   selected " + describe_offer(offer))

    if args.dry_run:
        print("\n[dry-run] would rent the offer above on image '{}'. "
              "Nothing was created.".format(args.image))
        return

    # 2. Ephemeral SSH key -----------------------------------------------------
    key_dir = tempfile.mkdtemp(prefix="vast_ssh_")
    key_path, public_key = generate_ssh_key(key_dir)

    # 3. Rent, measure, and always tear down -----------------------------------
    instance_id = api.create_instance(offer["id"], args.image, args.disk)
    print(">> created instance {} (${:.3f}/hr) -- billing has started"
          .format(instance_id, offer["dph_total"]))
    try:
        api.attach_ssh_key(instance_id, public_key)
        inst = wait_until_running(api, instance_id)
        host, port = inst["ssh_host"], inst["ssh_port"]
        print(">> instance running at {}:{}".format(host, port))

        wait_for_ssh(key_path, host, port)
        run_measurement(key_path, host, port, args.results_dir)

        summary = os.path.join(args.results_dir, "out", "quick_summary.csv")
        print("\n================ done ================")
        print("verbose results: {}".format(args.results_dir))
        if os.path.exists(summary):
            # Publish the one summary CSV to the canonical dataset location.
            os.makedirs(os.path.dirname(SUMMARY_DEST), exist_ok=True)
            shutil.copyfile(summary, SUMMARY_DEST)
            print("summary CSV -> {}".format(SUMMARY_DEST))
            with open(summary) as f:
                print(f.read())
    finally:
        if args.keep:
            print(">> --keep set: leaving instance {} RUNNING (still billing)"
                  .format(instance_id))
        else:
            print(">> destroying instance {}".format(instance_id))
            # The DELETE is what actually stops billing -- treat its own failure
            # as the alarm. The verify GET is best-effort and must not be able to
            # masquerade as a failed destroy.
            try:
                api.destroy_instance(instance_id)
            except Exception as exc:
                print(">> WARNING: destroy call failed ({}). Destroy instance {} "
                      "MANUALLY to stop billing.".format(exc, instance_id))
            else:
                try:
                    still_listed = api.get_instance(instance_id) is not None
                except Exception as exc:
                    print(">> instance {} destroy sent; could not verify ({}). "
                          "Confirm in the vast.ai console.".format(instance_id, exc))
                else:
                    if still_listed:
                        print(">> WARNING: instance {} still listed; verify in the "
                              "vast.ai console".format(instance_id))
                    else:
                        print(">> instance {} destroyed".format(instance_id))


if __name__ == "__main__":
    main()
