#!/usr/bin/env python3
"""Build the TurboVLA native QNN service on the isolated QCS8550 directory."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SOURCE = ROOT / "native/turbovla_qnn_server.cpp"


def run(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def ssh(host: str, command: str) -> None:
    run(["ssh", "-o", "BatchMode=yes", host, command])


def rsync(host: str, source: Path, destination: str) -> None:
    source_arg = f"{source}/" if source.is_dir() else str(source)
    run(["rsync", "-a", "--delete", source_arg, f"{host}:{destination}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", default="/opt/turbovla-qcs8550")
    parser.add_argument("--qairt-include", type=Path, default=os.environ.get("QAIRT_INCLUDE"))
    parser.add_argument("--runtime", default=os.environ.get("QAIRT_RUNTIME"))
    parser.add_argument("--port", type=int, default=10092)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    qairt_include = args.qairt_include / "QNN" if args.qairt_include and args.qairt_include.name != "QNN" else args.qairt_include
    if not SERVER_SOURCE.is_file() or not qairt_include or not qairt_include.is_dir():
        raise FileNotFoundError("native source or QAIRT 2.48 headers are missing")
    remote_native = f"{args.remote_root}/native"
    remote_include = f"{args.remote_root}/include"
    ssh(args.host, f"mkdir -p {shlex.quote(remote_native)} {shlex.quote(remote_include)}")
    rsync(args.host, SERVER_SOURCE, f"{remote_native}/turbovla_qnn_server.cpp")
    rsync(args.host, qairt_include, f"{remote_include}/QNN/")
    build = " ".join(
        [
            "g++ -std=c++17 -O2 -pipe -static-libstdc++ -static-libgcc",
            f"-I{shlex.quote(remote_include + '/QNN')}",
            shlex.quote(remote_native + "/turbovla_qnn_server.cpp"),
            "-ldl",
            "-o",
            shlex.quote(remote_native + "/turbovla_qnn_server"),
            "&& file",
            shlex.quote(remote_native + "/turbovla_qnn_server"),
            "&& sha256sum",
            shlex.quote(remote_native + "/turbovla_qnn_server"),
        ]
    )
    ssh(args.host, build)
    if args.start:
        if not args.runtime:
            raise ValueError("--runtime or QAIRT_RUNTIME is required with --start")
        runtime = args.runtime
        arm_lib = f"{runtime}/lib/aarch64-oe-linux-gcc11.2"
        dsp_lib = f"{runtime}/lib/hexagon-v73/unsigned"
        launch = " ".join(
            [
                f"if ss -ltn | grep -q {shlex.quote(':' + str(args.port) + ' ')}; then",
                f"echo port {args.port} is already in use >&2; exit 3; fi;",
                f"mkdir -p {shlex.quote(args.remote_root + '/logs')};",
                "nohup env",
                f"LD_LIBRARY_PATH={shlex.quote(arm_lib)}",
                f"ADSP_LIBRARY_PATH={shlex.quote(dsp_lib)}",
                f"DSP_LIBRARY_PATH={shlex.quote(dsp_lib)}",
                shlex.quote(remote_native + "/turbovla_qnn_server"),
                shlex.quote(args.remote_root),
                shlex.quote(runtime),
                str(args.port),
                ">",
                shlex.quote(args.remote_root + "/logs/turbovla_qnn_server.log"),
                "2>&1 < /dev/null &",
            ]
        )
        ssh(args.host, launch)
        ssh(args.host, f"sleep 1; cat {shlex.quote(args.remote_root + '/logs/turbovla_qnn_server.log')}")


if __name__ == "__main__":
    main()
