from __future__ import annotations

import sys

from gui.gui import main as gui_main
from gui.gui_runner import main as gui_runner_main


RUN_SPEC_FLAG = "--run-gui-spec"


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == RUN_SPEC_FLAG:
        sys.argv = [sys.argv[0], sys.argv[2], *sys.argv[3:]]
        return gui_runner_main()

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
