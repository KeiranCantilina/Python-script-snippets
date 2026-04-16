#!/usr/bin/env python3
"""
Serial‑USB ASCII float logger
----------------------------

* Reads lines that contain a single 6‑digit float expressed as ASCII
characters (e.g. "123.456\r\n").
* Prepends a timestamp and appends the pair to a CSV file.

Requirements:
 pip install pyserial
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import serial  # pyserial


# -------------------------- configuration -------------------------- #
SERIAL_PORT = "COM3"   # change to "COM3" on Windows, etc.
BAUDRATE = 115200
READ_TIMEOUT = 1.0            # seconds (None = block forever)

CSV_FILE = Path("data_log.csv")
#- #


def open_serial(port: str, baud: int, timeout: float) -> serial.Serial:
 """Open the serial port and return the Serial object."""
 try:
     ser = serial.Serial(port, baudrate=baud, timeout=timeout)
     print(f"Opened {ser.portstr} @ {baud} baud")
     return ser
 except serial.SerialException as exc:
     sys.exit(f"Could not open serial port: {exc}")


def init_csv(csv_path: Path):
 """Create CSV file with a header if it does not already exist."""
 already_exists = csv_path.is_file()
 fh = open(csv_path, "a", newline="", buffering=1)   # line‑buffered
 writer = csv.writer(fh)
 if not already_exists:
     writer.writerow(["timestamp", "value"])
 return fh, writer


def parse_ascii_float(line: bytes) -> float | None:
 """
 Convert a raw line (bytes) that contains an ASCII representation of a
 float to a Python float.

 Expected line format (including possible CR/LF): b'123.456\r\n'
 Returns None if the line cannot be parsed.
 """
 try:
     # Strip CR/LF and any surrounding whitespace, then decode ASCII.
     txt = line.strip().decode("ascii")
     return float(txt)          # Python will keep the full precision
 except (UnicodeDecodeError, ValueError):
     return None


def main():
 ser = open_serial(SERIAL_PORT, BAUDRATE, READ_TIMEOUT)
 csv_fh, csv_writer = init_csv(CSV_FILE)

 print("Logging started – press Ctrl‑C to stop.")
 try:
     while True:
         raw_line = ser.readline()          # reads up to newline
         if not raw_line:                    # timeout → nothing received
             continue

         value = parse_ascii_float(raw_line)
         if value is None:                   # malformed line – ignore
             continue

         # Timestamp with millisecond resolution
         ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
         csv_writer.writerow([ts, f"{value:.6f}"])
 except KeyboardInterrupt:
     print("\nInterrupted – closing files.")
 finally:
     ser.close()
     csv_fh.close()


if __name__ == "__main__":
 main()