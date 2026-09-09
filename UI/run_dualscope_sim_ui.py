import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "window" / "界面1220（可以使用版本+数据记录）双探子版_仿真版.py"
ROOT_DIR = MODULE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    spec = importlib.util.spec_from_file_location("dual_scope_sim_ui", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载界面文件: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
