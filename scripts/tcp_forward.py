#!/usr/bin/env python3
"""Small TCP forwarder for local OpenAI-compatible endpoints."""

from __future__ import annotations

import socket
import sys
import threading


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client: socket.socket, target_host: str, target_port: int) -> None:
    try:
        upstream = socket.create_connection((target_host, target_port), timeout=30)
    except OSError:
        client.close()
        return

    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: tcp_forward.py LISTEN_HOST LISTEN_PORT TARGET_HOST TARGET_PORT")
    listen_host = sys.argv[1]
    listen_port = int(sys.argv[2])
    target_host = sys.argv[3]
    target_port = int(sys.argv[4])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(256)
    print(f"[forward] {listen_host}:{listen_port} -> {target_host}:{target_port}", flush=True)

    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, target_host, target_port), daemon=True).start()


if __name__ == "__main__":
    main()
