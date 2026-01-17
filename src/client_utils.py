# src/client_utils.py
import sys
import time
from dataclasses import dataclass
from typing import Optional, Union

# --- Formatting Utils ---
def colorize(text: str, color: str) -> str:
    """Wraps text in ANSI color codes."""
    # 31=Red, 32=Green, 33=Yellow, 34=Blue, etc.
    return f"\033[{color}m{text}\033[0m"

def make_log_msg(level: str, msg: str) -> str:
    """Creates a formatted log message with timestamps and levels."""
    timestamp = time.strftime("%H:%M:%S")
    if level == "warning":
        prefix = colorize("[WARN]", "1;33")
    elif level == "info":
        prefix = colorize("[INFO]", "1;34")
    elif level == "error":
        prefix = colorize("[ERR ]", "1;31")
    elif level == "debug":
        prefix = colorize("[DBUG]", "1;32")
    else:
        prefix = f"[{level.upper()}]"
    
    return f"{colorize(timestamp, '90')} {prefix} {msg}"

# --- Simple Printer (for non-interactive terminals) ---
class RawPrinter:
    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def log(self, level: str, msg: str):
        print(make_log_msg(level, msg), file=sys.stderr)

    def print_token(self, token: str):
        self.stream.write(token)
        self.stream.flush()

# --- Advanced Printer (Moshi-style) ---
@dataclass
class LineEntry:
    msg: str
    color: Optional[str] = None

    def render(self) -> str:
        return colorize(self.msg, self.color) if self.color else self.msg

    def __len__(self) -> int:
        return len(self.msg)

class Line:
    """Manages the current line being printed (for streaming tokens)."""
    def __init__(self, stream):
        self.stream = stream
        self._entries: list[LineEntry] = []
        self._len = 0

    def add(self, msg: str, color: Optional[str] = None):
        entry = LineEntry(msg, color)
        self._entries.append(entry)
        self._len += len(msg)
        self.stream.write(entry.render())

    def clear(self):
        """Clears the current line using CR (Carriage Return)."""
        self.stream.write("\r" + " " * self._len + "\r")
        self._entries.clear()
        self._len = 0

    def restore(self):
        """Restores the cleared line."""
        for entry in self._entries:
            self.stream.write(entry.render())

class Printer:
    """
    Handles streaming output + Logging simultaneously.
    When a log comes in, it clears the current token stream line, 
    prints the log, and then puts the token stream back.
    """
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.line = Line(stream)

    def log(self, level: str, msg: str):
        # 1. Clear current streaming line
        self.line.clear()
        # 2. Print log message
        print(make_log_msg(level, msg), file=sys.stderr)
        # 3. Restore streaming line
        self.line.restore()
        self.stream.flush()

    def print_token(self, token: str, color: Optional[str] = None):
        self.line.add(token, color)
        self.stream.flush()

# --- Singleton Accessor ---
_printer: Optional[Union[Printer, RawPrinter]] = None

def get_logger():
    global _printer
    if _printer is None:
        # Check if we are in a real terminal
        if sys.stdout.isatty():
            _printer = Printer()
        else:
            _printer = RawPrinter()
    return _printer

def log(level: str, msg: str):
    get_logger().log(level, msg)