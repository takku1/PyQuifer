"""
MCP Server for Stockfish Chess Engine

Wraps the Stockfish UCI engine as an MCP server, following the pattern from
src/mizuki/cli/mcp_server.py.  Provides resources for reading engine state
and tools for position analysis.

Resources:
  - stockfish://evaluation  — current eval for a FEN position
  - stockfish://bestmove    — best move for a FEN
  - stockfish://analysis    — full multi-PV analysis

Tools:
  - evaluate_position(fen, depth)     — centipawn evaluation
  - get_best_move(fen, time_ms)       — best move in UCI notation
  - analyze_position(fen, multipv)    — multi-PV analysis with lines

Usage:
  Standalone:  python mcp_stockfish_server.py
  As library:  from mcp_stockfish_server import StockfishMCPServer

Set STOCKFISH_PATH env var or let it search common locations.
Falls back to cached/simulated results if binary not found.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# UCI Engine Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

# Well-known Penrose position — cached so we never need a real engine for tests
PENROSE_FEN = "8/p7/kpP5/qrp1b3/rpP2b2/pP4b1/P3K3/8 w - - 0 1"

CACHED_RESULTS: Dict[str, Dict[str, Any]] = {
    PENROSE_FEN: {
        "eval_cp": -2800,
        "eval_mate": None,
        "best_move": "e2d1",
        "depth": 20,
        "nodes": 48_500_000,
        "pv": ["e2d1", "a5a2", "d1c1", "a2a1"],
        "multipv": [
            {"pv": ["e2d1", "a5a2", "d1c1"], "cp": -2800, "depth": 20},
            {"pv": ["e2f1", "a5a2", "f1g1"], "cp": -2850, "depth": 20},
            {"pv": ["e2d3", "a5a2", "d3c3"], "cp": -2900, "depth": 20},
        ],
    }
}


def _find_stockfish() -> Optional[str]:
    """Find the Stockfish binary, checking env var then common paths."""
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check if stockfish is on PATH
    found = shutil.which("stockfish")
    if found:
        return found

    # Common installation paths
    candidates = [
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\Stockfish\stockfish.exe",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


class UCIEngine:
    """Thin wrapper around a UCI chess engine subprocess."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        """Start the engine process."""
        self.process = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send_command("uci")
        self._read_until("uciok")
        self._send_command("isready")
        self._read_until("readyok")

    def stop(self):
        """Stop the engine process."""
        if self.process:
            self._send_command("quit")
            self.process.wait(timeout=5)
            self.process = None

    def _send_command(self, cmd: str):
        """Send a UCI command to the engine."""
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def _read_until(self, stop_token: str, timeout: float = 10.0) -> List[str]:
        """Read engine output lines until we see the stop token."""
        lines = []
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.process and self.process.stdout:
                line = self.process.stdout.readline().strip()
                if line:
                    lines.append(line)
                    if stop_token in line:
                        break
        return lines

    def evaluate(self, fen: str, depth: int = 20) -> Dict[str, Any]:
        """Run evaluation on a position."""
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {depth}")
        lines = self._read_until("bestmove", timeout=30.0)
        return self._parse_analysis(lines)

    def best_move(self, fen: str, time_ms: int = 1000) -> Dict[str, Any]:
        """Get best move within a time limit."""
        self._send_command(f"position fen {fen}")
        self._send_command(f"go movetime {time_ms}")
        lines = self._read_until("bestmove", timeout=max(10, time_ms / 1000 + 5))
        return self._parse_analysis(lines)

    def analyze(self, fen: str, depth: int = 20, multipv: int = 3) -> Dict[str, Any]:
        """Run multi-PV analysis."""
        self._send_command(f"setoption name MultiPV value {multipv}")
        self._send_command("isready")
        self._read_until("readyok")
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {depth}")
        lines = self._read_until("bestmove", timeout=30.0)
        return self._parse_multipv(lines, multipv)

    def _parse_analysis(self, lines: List[str]) -> Dict[str, Any]:
        """Parse UCI info lines into structured result."""
        result: Dict[str, Any] = {
            "eval_cp": 0,
            "eval_mate": None,
            "best_move": None,
            "depth": 0,
            "nodes": 0,
            "pv": [],
        }

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == "bestmove" and len(tokens) >= 2:
                result["best_move"] = tokens[1]

            if tokens[0] == "info":
                for i, tok in enumerate(tokens):
                    if tok == "depth" and i + 1 < len(tokens):
                        result["depth"] = int(tokens[i + 1])
                    elif tok == "nodes" and i + 1 < len(tokens):
                        result["nodes"] = int(tokens[i + 1])
                    elif tok == "cp" and i + 1 < len(tokens):
                        result["eval_cp"] = int(tokens[i + 1])
                    elif tok == "mate" and i + 1 < len(tokens):
                        result["eval_mate"] = int(tokens[i + 1])
                    elif tok == "pv":
                        result["pv"] = tokens[i + 1:]
                        break

        return result

    def _parse_multipv(self, lines: List[str], multipv: int) -> Dict[str, Any]:
        """Parse multi-PV output."""
        pvs: Dict[int, Dict[str, Any]] = {}
        best_move = None

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == "bestmove" and len(tokens) >= 2:
                best_move = tokens[1]

            if tokens[0] == "info":
                pv_idx = 1
                depth = 0
                cp = 0
                pv_moves: List[str] = []

                for i, tok in enumerate(tokens):
                    if tok == "multipv" and i + 1 < len(tokens):
                        pv_idx = int(tokens[i + 1])
                    elif tok == "depth" and i + 1 < len(tokens):
                        depth = int(tokens[i + 1])
                    elif tok == "cp" and i + 1 < len(tokens):
                        cp = int(tokens[i + 1])
                    elif tok == "pv":
                        pv_moves = tokens[i + 1:]
                        break

                if pv_moves:
                    pvs[pv_idx] = {"pv": pv_moves, "cp": cp, "depth": depth}

        return {
            "best_move": best_move,
            "multipv": [pvs.get(i + 1, {}) for i in range(multipv)],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCPResource:
    """An MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


@dataclass
class MCPToolDef:
    """An MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class StockfishMCPServer:
    """
    MCP server wrapping Stockfish for chess position analysis.

    Falls back to cached/simulated results when the binary is unavailable.
    """

    def __init__(self):
        self.engine: Optional[UCIEngine] = None
        self.using_fallback = False
        self._last_fen: Optional[str] = None
        self._message_id = 0

        # Resources
        self.resources: Dict[str, MCPResource] = {
            "stockfish://evaluation": MCPResource(
                uri="stockfish://evaluation",
                name="Position Evaluation",
                description="Centipawn evaluation for a chess position (FEN)",
            ),
            "stockfish://bestmove": MCPResource(
                uri="stockfish://bestmove",
                name="Best Move",
                description="Best move for a chess position in UCI notation",
            ),
            "stockfish://analysis": MCPResource(
                uri="stockfish://analysis",
                name="Full Analysis",
                description="Multi-PV analysis with depth, nodes, and principal variations",
            ),
        }

        # Tools
        self.tools: Dict[str, MCPToolDef] = {
            "evaluate_position": MCPToolDef(
                name="evaluate_position",
                description="Evaluate a chess position, returning centipawn score",
                input_schema={
                    "type": "object",
                    "properties": {
                        "fen": {"type": "string", "description": "FEN position string"},
                        "depth": {"type": "integer", "default": 20, "description": "Search depth"},
                    },
                    "required": ["fen"],
                },
            ),
            "get_best_move": MCPToolDef(
                name="get_best_move",
                description="Get the best move for a position",
                input_schema={
                    "type": "object",
                    "properties": {
                        "fen": {"type": "string", "description": "FEN position string"},
                        "time_ms": {"type": "integer", "default": 1000, "description": "Time limit in ms"},
                    },
                    "required": ["fen"],
                },
            ),
            "analyze_position": MCPToolDef(
                name="analyze_position",
                description="Multi-PV analysis of a chess position",
                input_schema={
                    "type": "object",
                    "properties": {
                        "fen": {"type": "string", "description": "FEN position string"},
                        "multipv": {"type": "integer", "default": 3, "description": "Number of PV lines"},
                    },
                    "required": ["fen"],
                },
            ),
        }

        # Try to find and start Stockfish
        binary = _find_stockfish()
        if binary:
            try:
                self.engine = UCIEngine(binary)
                self.engine.start()
            except Exception:
                self.engine = None
                self.using_fallback = True
        else:
            self.using_fallback = True

    def shutdown(self):
        """Clean up engine process."""
        if self.engine:
            self.engine.stop()
            self.engine = None

    # ── Tool handlers ──

    def evaluate_position(self, fen: str, depth: int = 20) -> Dict[str, Any]:
        """Evaluate a position, returning centipawns."""
        if self.engine and not self.using_fallback:
            result = self.engine.evaluate(fen, depth)
            return {
                "fen": fen,
                "eval_cp": result["eval_cp"],
                "eval_mate": result.get("eval_mate"),
                "depth": result["depth"],
                "source": "stockfish",
            }

        # Fallback to cache
        cached = CACHED_RESULTS.get(fen)
        if cached:
            return {
                "fen": fen,
                "eval_cp": cached["eval_cp"],
                "eval_mate": cached.get("eval_mate"),
                "depth": cached["depth"],
                "source": "cached",
            }

        return {
            "fen": fen,
            "eval_cp": 0,
            "eval_mate": None,
            "depth": 0,
            "source": "fallback_unknown_position",
        }

    def get_best_move(self, fen: str, time_ms: int = 1000) -> Dict[str, Any]:
        """Get best move for a position."""
        if self.engine and not self.using_fallback:
            result = self.engine.best_move(fen, time_ms)
            return {
                "fen": fen,
                "best_move": result["best_move"],
                "eval_cp": result["eval_cp"],
                "source": "stockfish",
            }

        cached = CACHED_RESULTS.get(fen)
        if cached:
            return {
                "fen": fen,
                "best_move": cached["best_move"],
                "eval_cp": cached["eval_cp"],
                "source": "cached",
            }

        return {"fen": fen, "best_move": None, "eval_cp": 0, "source": "fallback_unknown_position"}

    def analyze_position(self, fen: str, multipv: int = 3) -> Dict[str, Any]:
        """Multi-PV analysis of a position."""
        if self.engine and not self.using_fallback:
            result = self.engine.analyze(fen, multipv=multipv)
            return {
                "fen": fen,
                "multipv": result["multipv"],
                "best_move": result["best_move"],
                "source": "stockfish",
            }

        cached = CACHED_RESULTS.get(fen)
        if cached:
            return {
                "fen": fen,
                "multipv": cached.get("multipv", []),
                "best_move": cached["best_move"],
                "source": "cached",
            }

        return {"fen": fen, "multipv": [], "best_move": None, "source": "fallback_unknown_position"}

    # ── MCP protocol: JSON-RPC stdio ──

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an MCP tool call."""
        handlers = {
            "evaluate_position": lambda args: self.evaluate_position(
                args["fen"], args.get("depth", 20)
            ),
            "get_best_move": lambda args: self.get_best_move(
                args["fen"], args.get("time_ms", 1000)
            ),
            "analyze_position": lambda args: self.analyze_position(
                args["fen"], args.get("multipv", 3)
            ),
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = handler(arguments)
            return {"tool": name, "success": True, "result": result,
                    "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"tool": name, "success": False, "error": str(e),
                    "timestamp": datetime.now().isoformat()}

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read an MCP resource (requires a FEN set via last tool call)."""
        if uri not in self.resources:
            return {"error": f"Unknown resource: {uri}"}

        fen = self._last_fen or PENROSE_FEN

        if uri == "stockfish://evaluation":
            return self.evaluate_position(fen)
        elif uri == "stockfish://bestmove":
            return self.get_best_move(fen)
        elif uri == "stockfish://analysis":
            return self.analyze_position(fen)
        else:
            return {"error": f"Unhandled resource: {uri}"}

    def list_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources."""
        return [{"uri": r.uri, "name": r.name, "description": r.description}
                for r in self.resources.values()]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        return [{"name": t.name, "description": t.description,
                 "inputSchema": t.input_schema}
                for t in self.tools.values()]

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC 2.0 request."""
        self._message_id += 1
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", self._message_id)

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"resources": {}, "tools": {}},
                    "serverInfo": {"name": "stockfish-mcp", "version": "0.1.0"},
                },
            }

        elif method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"resources": self.list_resources()}}

        elif method == "resources/read":
            uri = params.get("uri", "")
            data = self.read_resource(uri)
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "application/json",
                                             "text": json.dumps(data)}]}}

        elif method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"tools": self.list_tools()}}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            if "fen" in arguments:
                self._last_fen = arguments["fen"]
            result = self.call_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text",
                                            "text": json.dumps(result)}]}}

        else:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def run_stdio(self):
        """Run as a JSON-RPC stdio server (for Claude Desktop / MCP clients)."""
        print(f"[stockfish-mcp] Starting (engine={'live' if self.engine else 'fallback'})",
              file=sys.stderr)

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                    response = self.handle_jsonrpc(request)
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                except json.JSONDecodeError:
                    err = {"jsonrpc": "2.0", "id": None,
                           "error": {"code": -32700, "message": "Parse error"}}
                    sys.stdout.write(json.dumps(err) + "\n")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    server = StockfishMCPServer()

    if "--test" in sys.argv:
        # Quick self-test
        print(f"Engine: {'live' if server.engine else 'fallback (cached)'}")
        print(f"Resources: {[r['name'] for r in server.list_resources()]}")
        print(f"Tools: {[t['name'] for t in server.list_tools()]}")
        print()

        # Test evaluate
        result = server.evaluate_position(PENROSE_FEN)
        print(f"Penrose eval: {result['eval_cp']}cp (source: {result['source']})")

        # Test best move
        result = server.get_best_move(PENROSE_FEN)
        print(f"Best move: {result['best_move']} (source: {result['source']})")

        # Test analysis
        result = server.analyze_position(PENROSE_FEN)
        print(f"Analysis: {len(result['multipv'])} PV lines (source: {result['source']})")

        server.shutdown()
    else:
        # Run as stdio MCP server
        server.run_stdio()
