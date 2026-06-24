#!/usr/bin/env python3
"""生成模块级测试覆盖基线报告。

由于 coverage 工具当前无法安装，此脚本通过分析测试文件中的 import 和 mock
patch 目标，推断哪些源码模块被测试直接引用。结果作为模块级覆盖基线，
用于识别未测试的盲点。"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def find_imports_in_test(test_file: Path) -> set[str]:
    """解析测试文件，提取所有 florence_forge 相关的 import / patch 引用。"""
    refs: set[str] = set()
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return refs

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("florence_forge"):
                    refs.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("florence_forge"):
                refs.add(module)
                for alias in node.names:
                    refs.add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Call):
            # 检测 unittest.mock.patch 目标字符串
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("patch", "patch.object"):
                for kw in node.keywords:
                    if kw.arg == "target" and isinstance(kw.value, ast.Constant):
                        target = str(kw.value.value)
                        if target.startswith("florence_forge"):
                            refs.add(target)
    return refs


def main():
    root = Path("/Users/gatilin/PycharmProjects/FlorenceForge")
    src_dir = root / "florence_forge"
    test_dir = root / "tests"

    # 收集所有源文件（排除 __pycache__）
    src_files = [
        f.relative_to(src_dir)
        for f in src_dir.rglob("*.py")
        if "__pycache__" not in f.parts and f.name != "__init__.py"
    ]
    # 映射为模块名
    src_modules = {
        str(f.with_suffix("")).replace("/", "."): f
        for f in src_files
    }

    # 收集测试引用
    test_refs: set[str] = set()
    for test_file in test_dir.rglob("*.py"):
        if "__pycache__" in test_file.parts:
            continue
        test_refs |= find_imports_in_test(test_file)

    # 判断每个源模块是否被测试引用
    covered = set()
    uncovered = set()
    for mod_name, rel_path in src_modules.items():
        # 检查是否有测试直接或间接引用该模块
        is_covered = any(
            ref == f"florence_forge.{mod_name}" or ref.startswith(f"florence_forge.{mod_name}.")
            for ref in test_refs
        )
        if is_covered:
            covered.add(mod_name)
        else:
            uncovered.add(mod_name)

    # 统计
    total = len(src_modules)
    cover_pct = len(covered) / total * 100 if total else 0

    report = {
        "total_modules": total,
        "covered_modules": len(covered),
        "uncovered_modules": len(uncovered),
        "module_coverage_percent": round(cover_pct, 2),
        "covered": sorted(covered),
        "uncovered": sorted(uncovered),
    }

    output = root / "coverage_baseline.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📊 模块级测试覆盖基线: {cover_pct:.1f}% ({len(covered)}/{total})")
    print(f"   未覆盖模块: {len(uncovered)}")
    if uncovered:
        for mod in sorted(uncovered)[:15]:
            print(f"      - {mod}")
    print(f"\n💾 报告已保存: {output}")


if __name__ == "__main__":
    main()
