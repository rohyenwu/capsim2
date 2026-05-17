#!/usr/bin/env python
"""Dependency-free web backend for WiFi v9 BEB vs RL Mbps simulations."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
SHIM_DIR = APP_DIR / "shims"
REPO_DIR = Path(os.environ.get("MAPPO_REPO_DIR", APP_DIR.parent / "mappo")).resolve()
MODEL_DIR = (
    Path(os.environ.get("MAPPO_MODEL_DIR", ""))
    if os.environ.get("MAPPO_MODEL_DIR")
    else REPO_DIR
    / "model"
    / "WiFi_v9"
    / "mappo"
    / "wifi_v9_train_airtime50ms_m15m25_s3s5_parallel_vec4_d2lt_mldsucc1_sld07_10_ntop1_cidle03_1600k_lr1e4_ent5e3_seed1"
)

MAX_MLD = 30
MAX_SLD = 10
COMBOS = [
    {"id": f"m{mld}_s{sld}", "mld": mld, "sld": sld}
    for sld in (2, 4, 6)
    for mld in (10, 15, 20, 25, 30)
]
VALID_DURATIONS = {10, 30, 50}
FIXED_SEED = int(os.environ.get("CAPSIM_SEED", "1"))

_event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
_run_lock = threading.Lock()
_running = False


RE_EPISODE = re.compile(
    r"\[(?P<policy>RL|BEB) Mbps Eval\] Episode (?P<ep>\d+)/(?P<total>\d+)"
    r" \| mbps/system=(?P<mbps_sys>[0-9.]+)"
    r" \| mbps/mld_total=(?P<mbps_mld>[0-9.]+)"
    r" \| mbps/sld_total=(?P<mbps_sld>[0-9.]+)"
    r" \| tx_ratio=(?P<tx>[0-9.]+)"
)
RE_SUMMARY_HDR = re.compile(r"\[(RL|BEB) Mbps Summary\]")
RE_SUMMARY_LINE = re.compile(r"^\s{2}(?P<key>[\w/.]+):\s+(?P<val>[0-9.eE+\-]+)")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _combo_by_id(combo_id: str) -> dict[str, int | str] | None:
    return next((combo for combo in COMBOS if combo["id"] == combo_id), None)


def _base_args(combo: dict[str, int | str], duration_sec: int, seed: int) -> list[str]:
    return [
        "--env_name",
        "WiFi_v9",
        "--num_mld",
        str(combo["mld"]),
        "--num_sld",
        str(combo["sld"]),
        "--max_mld",
        str(MAX_MLD),
        "--max_sld",
        str(MAX_SLD),
        "--round_length",
        "500",
        "--mu_min",
        "0.01",
        "--mu_max",
        "0.12",
        "--eta",
        "0.2",
        "--zeta",
        "0.2",
        "--c_idle",
        "0.3",
        "--collision_penalty",
        "1.0",
        "--non_top_tx_penalty",
        "1.0",
        "--theta_scale",
        "1.0",
        "--sld_target_low_scale",
        "0.7",
        "--sld_target_high_scale",
        "1.0",
        "--sld_target_bonus",
        "0.5",
        "--mld_success_reward",
        "1.0",
        "--eval_episodes",
        "1",
        "--eval_duration_sec",
        str(float(duration_sec)),
        "--slot_time_sec",
        "9e-6",
        "--debug_prob_steps",
        "0",
        "--seed",
        str(seed),
        "--use_wandb",
    ]


def _build_cmd(policy: str, combo: dict[str, int | str], duration_sec: int, seed: int) -> list[str]:
    module = (
        "onpolicy.scripts.eval.eval_wifi_v9_rl_mbps"
        if policy == "rl"
        else "onpolicy.scripts.eval.eval_wifi_v9_beb_mbps"
    )
    args = [
        sys.executable,
        "-m",
        module,
        "--algorithm_name",
        "mappo",
        "--experiment_name",
        f"cap_sim_{policy}_{combo['id']}_{duration_sec}s",
        *_base_args(combo, duration_sec, seed),
    ]
    if policy == "rl":
        args.extend(["--stochastic", "--model_dir", str(MODEL_DIR)])
    return args


def _stream_proc(proc: subprocess.Popen[str], policy: str, eq: queue.Queue[dict[str, Any]]) -> None:
    in_summary = False
    summary: dict[str, float] = {}

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        eq.put({"type": "log", "policy": policy, "line": line})

        episode_match = RE_EPISODE.search(line)
        if episode_match:
            eq.put(
                {
                    "type": "episode",
                    "policy": policy,
                    "ep": int(episode_match.group("ep")),
                    "total": int(episode_match.group("total")),
                    "mbps_system": float(episode_match.group("mbps_sys")),
                    "mbps_mld": float(episode_match.group("mbps_mld")),
                    "mbps_sld": float(episode_match.group("mbps_sld")),
                    "tx_ratio": float(episode_match.group("tx")),
                }
            )
            continue

        if RE_SUMMARY_HDR.search(line):
            in_summary = True
            continue

        if in_summary:
            summary_match = RE_SUMMARY_LINE.match(line)
            if summary_match:
                summary[summary_match.group("key")] = float(summary_match.group("val"))
            else:
                in_summary = False

    proc.wait()
    if summary:
        eq.put(
            {
                "type": "summary",
                "policy": policy,
                "returncode": proc.returncode,
                "mbps_system": summary.get("mbps/system", 0.0),
                "mbps_mld": summary.get("mbps/mld_total", 0.0),
                "mbps_sld": summary.get("mbps/sld_total", 0.0),
                "mbps_24_mld": summary.get("mbps/2_4GHz/mld", 0.0),
                "mbps_24_sld": summary.get("mbps/2_4GHz/sld", 0.0),
                "mbps_5_mld": summary.get("mbps/5GHz/mld", 0.0),
                "success_rate": summary.get("success_rate/system_per_event", 0.0),
                "collision_rate": summary.get("collision_rate/system_per_event", 0.0),
                "tx_ratio": summary.get("action/transmit_ratio", 0.0),
                "active_mld": summary.get("scenario/active_mld", 0.0),
                "active_sld": summary.get("scenario/active_sld", 0.0),
            }
        )
    else:
        eq.put(
            {
                "type": "error",
                "policy": policy,
                "msg": f"{policy.upper()} process finished with rc={proc.returncode}, but no summary was found.",
                "returncode": proc.returncode,
            }
        )


def _start_run(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    global _event_queue, _running

    combo_id = str(payload.get("combo", "m15_s4"))
    duration_sec = int(payload.get("duration_sec", 30))
    combo = _combo_by_id(combo_id)

    if combo is None:
        return 400, {"error": f"Unknown combo: {combo_id}"}
    if duration_sec not in VALID_DURATIONS:
        return 400, {"error": "duration_sec must be one of 10, 30, 50"}
    if not REPO_DIR.exists():
        return 500, {"error": f"Repo directory was not found: {REPO_DIR}"}
    if not MODEL_DIR.exists():
        return 500, {"error": f"RL model directory was not found: {MODEL_DIR}"}

    with _run_lock:
        if _running:
            return 409, {"error": "A simulation is already running."}
        _running = True
        _event_queue = queue.Queue()
        eq = _event_queue

    seed = FIXED_SEED
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SHIM_DIR), str(REPO_DIR), env.get("PYTHONPATH", "")]
    )
    env["PYTHONUNBUFFERED"] = "1"

    def launch() -> None:
        global _running
        try:
            eq.put({"type": "started", "combo": combo, "duration_sec": duration_sec, "seed": seed})
            procs: dict[str, subprocess.Popen[str]] = {}
            for policy in ("beb", "rl"):
                procs[policy] = subprocess.Popen(
                    _build_cmd(policy, combo, duration_sec, seed),
                    cwd=str(REPO_DIR),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

            threads = [
                threading.Thread(target=_stream_proc, args=(procs[policy], policy, eq), daemon=True)
                for policy in ("beb", "rl")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except Exception as exc:
            eq.put({"type": "error", "policy": "server", "msg": str(exc)})
        finally:
            eq.put({"type": "done"})
            with _run_lock:
                _running = False

    threading.Thread(target=launch, daemon=True).start()
    return 200, {"status": "started", "combo": combo, "duration_sec": duration_sec, "seed": seed}


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CapSimHTTP/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(APP_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/options":
            self._send_json(
                200,
                {
                    "combos": COMBOS,
                    "durations": sorted(VALID_DURATIONS),
                    "model_exists": MODEL_DIR.exists(),
                    "repo_dir": str(REPO_DIR),
                },
            )
        elif self.path == "/api/status":
            self._send_json(200, {"running": _running})
        elif self.path == "/api/progress":
            self._send_sse()
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/start":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        status, body = _start_run(payload)
        self._send_json(status, body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_headers(self, status: int, content_type: str, content_length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _send_json(self, status: int, payload: Any) -> None:
        data = _json_bytes(payload)
        self._send_headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "File not found"})
            return
        data = path.read_bytes()
        self._send_headers(200, content_type, len(data))
        self.wfile.write(data)

    def _send_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b": connected\n\n")
        self.wfile.flush()

        while True:
            try:
                event = _event_queue.get(timeout=30)
            except queue.Empty:
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                continue

            data = _json_bytes(event)
            self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.flush()
            if event.get("type") == "done":
                break


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    server = ThreadingHTTPServer(("0.0.0.0", port), DemoHandler)
    print(f"WiFi v9 simulation demo: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
