"""Compatibility entry point for TDCR Vision Studio.

The product UI is split by responsibility under ``workstation``:

- ``state.py``: single workflow state;
- ``worker.py``: background processing boundary;
- ``widgets.py``: reusable visual components;
- ``pages.py``: focused workflow pages;
- ``window.py``: navigation and state transitions.

Keeping this tiny module preserves ``python -m Visual_information.seven_marker_3d_fusion.ui`` and
the existing Windows launcher without retaining a second UI implementation.
"""

from .workstation.window import WorkstationWindow, main

OfflineFusionWindow = WorkstationWindow

__all__ = ["OfflineFusionWindow", "WorkstationWindow", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
