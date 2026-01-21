"""
Hive-Reflex 2.1 Pytest 配置
"""

import pytest
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'imc22_sdk' / 'python'))
sys.path.insert(0, str(PROJECT_ROOT / 'mlir_compiler'))
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--hil", action="store_true", default=False,
        help="启用 HIL (硬件在环) 测试"
    )
    parser.addoption(
        "--port", action="store", default="COM3",
        help="硬件串口"
    )
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="运行慢速测试"
    )


def pytest_configure(config):
    """配置 Pytest 标记"""
    config.addinivalue_line(
        "markers", "hil: 硬件在环测试 (需要 --hil 选项)"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试 (需要 --slow 选项)"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    if not config.getoption("--hil"):
        skip_hil = pytest.mark.skip(reason="需要 --hil 选项")
        for item in items:
            if "hil" in item.keywords or "HIL" in item.name:
                item.add_marker(skip_hil)
    
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="需要 --slow 选项")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
