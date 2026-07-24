#!/usr/bin/env python3
"""
Convert the raw llama8Bshort Ramulator command traces into the format the BRAM
trace runner (HBMTraceRunnerBRAM / StreamProgram::load_csv_entries) expects, and
guarantee every bank is precharged at the end of each trace so it loops cleanly.

Why this is needed
------------------
The raw traces (cmd_traces/rebuttal/llama8Bshort/*.csv) are in Ramulator's
13-column, comma-space-separated format:

    clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr, pim_cmd, operand, req_type

The runner instead parses the 8-column trace-runner format *positionally* and
matches command names with an exact string compare:

    timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id

Fed the raw files directly, the runner (a) sees " ACT"/" RD" (leading space) which
never matches "ACT"/"RD", so every row becomes a NOP and the trace loads empty,
and (b) would read the address fields one column off because of the extra `rank`
column. This script fixes both: it drops `rank` (+ the trailing addr/pim/operand/
req_type columns), strips whitespace, and re-emits the 8-column format.

Closing precharges
------------------
The replay loops the trace forever, so the last iteration's open rows must be
closed before the next iteration's opening ACTs. We replay each trace tracking
which (pc, bankgroup, bank) is left open (ACT opens; PRE/PREA/RDA/WRA close;
RD/WR/REF leave open) and append a PRE for each still-open bank after the last
command. (run_on_infra_bram.sh additionally appends --tail-gap idle slots so the
loop seam leaves >= tRP before the re-opening ACTs.)

Two output sets are produced for every input trace:
  * "common" (DST_DIR): one flat runner CSV keeping BOTH pseudo-channels in the
    same file (pc0 and pc1 replayed interleaved, exactly as recorded).
  * per-PC (DST_DIR_pc0 / DST_DIR_pc1): the same trace with only the commands
    that target that single pseudo-channel kept. Timestamps are preserved, so
    the other PC's slots simply become idle NOPs (same wall-clock schedule, just
    one PC exercised). Closing PREs are appended per set.

Usage:
    python3 convert_llama8Bshort.py [SRC_DIR] [DST_DIR] [--max-bank-depth N]

    SRC_DIR  default cmd_traces/rebuttal/llama8Bshort
    DST_DIR  default cmd_traces/rebuttal/llama8Bshort_runner
             (the per-PC sets go to DST_DIR_pc0 and DST_DIR_pc1)
    --max-bank-depth N   also truncate each set so no replay bank
                         (timestamp mod 4) exceeds N entries (room is reserved
                         for the closing PREs). Omit to keep traces full length;
                         the current design's BANK_DEPTH is 131072.
"""
import argparse
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 8-column trace-runner format (positional order parsed by load_csv_entries).
HEADER = ["timestamp", "command", "channel_id", "pseudochannel_id",
          "bankgroup_id", "bank_id", "row_id", "column_id"]

# Commands for which the column address is meaningful (others get a blank column).
COL_CMDS = {"RD", "RDA", "WR", "WRA"}

# Encoder bit-width limits (from SoftMCPlatform::encodeCommand).
LIMITS = {"channel_id": 15, "pseudochannel_id": 1, "bankgroup_id": 3,
          "bank_id": 3, "row_id": 16383, "column_id": 31}

# Slot spacing between the closing PRE commands appended at the end of a trace.
# A multiple of 4 keeps them all in one replay bank (bank = timestamp mod 4).
PRE_GAP = 4


def normalize_header(fieldnames):
    """Map stripped/lower-cased raw header names -> their position-preserving key.

    The raw header is e.g. 'clk, CMD, channel, pseudochannel, rank, ...' (note the
    spaces and the upper-case CMD). We look fields up case-insensitively and
    whitespace-insensitively so the parser is robust to those quirks.
    """
    norm = {}
    for name in fieldnames:
        if name is None:
            continue
        norm[name.strip().lower()] = name
    return norm


def load_raw(src_path):
    """Read a raw llama8Bshort trace, return (rows, max_ts).

    rows is a list of 8-column runner-format dicts in input (timestamp) order.
    """
    rows = []
    max_ts = 0
    warnings = set()
    with open(src_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name.strip().lower(): i for i, name in enumerate(header)}

        def col(parts, key):
            return parts[idx[key]].strip()

        for parts in reader:
            if not parts:
                continue
            cmd = col(parts, "cmd").upper()
            if not cmd:
                continue
            ts = int(col(parts, "clk"))
            out = {
                "timestamp": ts,
                "command": cmd,
                "channel_id": col(parts, "channel"),
                "pseudochannel_id": col(parts, "pseudochannel"),
                "bankgroup_id": col(parts, "bankgroup"),
                "bank_id": col(parts, "bank"),
                "row_id": col(parts, "row"),
                "column_id": col(parts, "column") if cmd in COL_CMDS else "",
            }
            for field, limit in LIMITS.items():
                v = out[field]
                if v != "" and int(v) > limit:
                    warnings.add(f"{field}={v} exceeds encoder limit {limit}")
            rows.append(out)
            if ts > max_ts:
                max_ts = ts
    return rows, max_ts, warnings


def truncate_to_bank_depth(rows, bank_depth, reserve=64):
    """Keep the longest leading (timestamp-ordered) prefix that fits the replay
    engine: each bank = timestamp mod 4 must hold <= bank_depth entries. `reserve`
    holds back room in every bank for the closing PREs appended afterwards.
    Returns (kept_rows, dropped_count).
    """
    limit = bank_depth - reserve
    counts = [0, 0, 0, 0]
    keep = 0
    for i, r in enumerate(rows):
        b = r["timestamp"] % 4
        if counts[b] + 1 > limit:
            break
        counts[b] += 1
        keep = i + 1
    return rows[:keep], len(rows) - keep


def append_closing_pres(rows):
    """Append a PRE for every (pc, bankgroup, bank) still open after the last
    command, so the trace ends with all banks precharged. Returns the PRE rows
    (empty if nothing is left open). PREs are spaced PRE_GAP after the trace's
    max timestamp.
    """
    open_banks = {}   # (pc, bg, bank) -> (channel, row last activated)
    max_ts = 0
    for r in rows:
        if r["timestamp"] > max_ts:
            max_ts = r["timestamp"]
        key = (r["pseudochannel_id"], r["bankgroup_id"], r["bank_id"])
        c = r["command"]
        if c == "ACT":
            open_banks[key] = (r["channel_id"], r["row_id"])
        elif c in ("PRE", "RDA", "WRA"):
            open_banks.pop(key, None)
        elif c == "PREA":
            open_banks.clear()
        # RD / WR / REF leave the row open

    pres = []
    ts = max_ts
    for (pc, bg, bank) in sorted(open_banks,
                                 key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        ch, row = open_banks[(pc, bg, bank)]
        ts += PRE_GAP
        pres.append({
            "timestamp": ts,
            "command": "PRE",
            "channel_id": ch,
            "pseudochannel_id": pc,
            "bankgroup_id": bg,
            "bank_id": bank,
            "row_id": row,      # PRE ignores the row; kept for clarity
            "column_id": "",
        })
    return pres


def write_runner_csv(dst_path, rows):
    with open(dst_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r[k] for k in HEADER])


def filter_pc(rows, pc):
    """Keep only the commands targeting pseudo-channel `pc` (timestamps unchanged)."""
    target = str(pc)
    return [r for r in rows if str(r["pseudochannel_id"]).strip() == target]


def finalize(rows, max_bank_depth):
    """Optionally truncate to bank depth, then append closing PREs.

    Returns (rows_out, dropped, n_pres, max_bank).
    """
    dropped = 0
    if max_bank_depth is not None:
        rows, dropped = truncate_to_bank_depth(rows, max_bank_depth)
    pres = append_closing_pres(rows)
    rows = rows + pres
    counts = [0, 0, 0, 0]
    for r in rows:
        counts[r["timestamp"] % 4] += 1
    return rows, dropped, len(pres), max(counts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir", nargs="?",
                    default=os.path.join(HERE, "rebuttal", "llama8Bshort"),
                    help="directory of raw llama8Bshort traces")
    ap.add_argument("dst_dir", nargs="?",
                    default=os.path.join(HERE, "rebuttal", "llama8Bshort_runner"),
                    help="output directory for the common (both-PC) runner traces")
    ap.add_argument("--max-bank-depth", type=int, default=None, metavar="N",
                    help="truncate each set so no replay bank exceeds N entries")
    args = ap.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.src_dir, "*.csv")))
    if not src_files:
        ap.error(f"no .csv traces found in {args.src_dir}")

    # One "common" set (both PCs) + one isolated set per pseudo-channel.
    PCS = [0, 1]
    out_dirs = {"common": args.dst_dir}
    for pc in PCS:
        out_dirs[f"pc{pc}"] = f"{args.dst_dir}_pc{pc}"
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"Converting {len(src_files)} trace(s) from {args.src_dir}")
    for label, d in out_dirs.items():
        print(f"  {label:7} -> {d}")
    if args.max_bank_depth is not None:
        print(f"  truncating each replay bank to <= {args.max_bank_depth} entries")
    print(f"  {'trace':18} {'set':7} {'cmds_in':>9} {'dropped':>8} {'pres':>5} {'cmds_out':>9} {'maxbank':>8}")

    for src in src_files:
        name = os.path.basename(src)
        rows, _max_ts, warnings = load_raw(src)

        # set_label -> rows targeting it (common = all PCs)
        sets = {"common": rows}
        for pc in PCS:
            sets[f"pc{pc}"] = filter_pc(rows, pc)

        for label, set_rows in sets.items():
            n_in = len(set_rows)
            out_rows, dropped, n_pres, max_bank = finalize(set_rows, args.max_bank_depth)
            write_runner_csv(os.path.join(out_dirs[label], name), out_rows)
            print(f"  {name:18} {label:7} {n_in:>9} {dropped:>8} {n_pres:>5} "
                  f"{len(out_rows):>9} {max_bank:>8}")
        for w in sorted(warnings):
            print(f"      WARNING [{name}]: {w}")

    print("Done.")


if __name__ == "__main__":
    main()
