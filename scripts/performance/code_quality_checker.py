#!/usr/bin/env python3
"""
代码质量检查器 - 分析FlorenceForge框架的代码质量
"""

import sys
import ast
import json
import time
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import Counter

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class CodeIssue:
    """代码问题数据类"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None

@dataclass
class CodeMetrics:
    """代码指标数据类"""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    functions: int = 0
    classes: int = 0
    complexity: int = 0
    imports: int = 0
    docstring_coverage: float = 0.0
    
@dataclass
class QualityReport:
    """质量报告数据类"""
    timestamp: str
    total_files: int
    total_issues: int
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    issues: List[CodeIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class CodeQualityChecker:
    """代码质量检查器"""
    
    def __init__(self, output_dir: str = "./code_quality_results"):
        """初始化代码质量检查器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.issues = []
        self.metrics = CodeMetrics()
        self.file_count = 0
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "code_quality_checker.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 代码风格规则
        self.style_rules = {
            'max_line_length': 100,
            'max_function_length': 50,
            'max_class_length': 200,
            'max_complexity': 10,
            'require_docstrings': True,
            'naming_conventions': {
                'function': r'^[a-z_][a-z0-9_]*$',
                'class': r'^[A-Z][a-zA-Z0-9]*$',
                'constant': r'^[A-Z_][A-Z0-9_]*$',
                'variable': r'^[a-z_][a-z0-9_]*$'
            }
        }
        
        self.logger.info("代码质量检查器初始化完成")
    
    def add_issue(self, file_path: str, line_number: int, issue_type: str, 
                  severity: str, message: str, suggestion: Optional[str] = None):
        """添加代码问题"""
        issue = CodeIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion
        )
        self.issues.append(issue)
    
    def check_file(self, file_path: Path) -> None:
        """检查单个文件"""
        if not file_path.suffix == '.py':
            return
        
        self.logger.debug(f"检查文件: {file_path}")
        self.file_count += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # 解析AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.add_issue(
                    str(file_path), e.lineno or 0, "syntax_error", "error",
                    f"语法错误: {e.msg}", "修复语法错误"
                )
                return
            
            # 执行各种检查
            self._check_line_metrics(file_path, lines)
            self._check_line_length(file_path, lines)
            self._check_imports(file_path, tree)
            self._check_functions(file_path, tree)
            self._check_classes(file_path, tree)
            self._check_naming_conventions(file_path, tree)
            self._check_docstrings(file_path, tree)
            self._check_complexity(file_path, tree)
            self._check_code_smells(file_path, tree, lines)
            
        except Exception as e:
            self.logger.error(f"检查文件 {file_path} 时出错: {e}")
            self.add_issue(
                str(file_path), 0, "check_error", "error",
                f"文件检查失败: {str(e)}"
            )
    
    def _check_line_metrics(self, file_path: Path, lines: List[str]) -> None:
        """检查行指标"""
        total_lines = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
        
        # 更新全局指标
        self.metrics.total_lines += total_lines
        self.metrics.code_lines += code_lines
        self.metrics.comment_lines += comment_lines
        self.metrics.blank_lines += blank_lines
        
        # 检查注释比例
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            if comment_ratio < 0.1:  # 注释少于10%
                self.add_issue(
                    str(file_path), 0, "low_comment_ratio", "warning",
                    f"注释比例过低: {comment_ratio:.1%}",
                    "增加代码注释以提高可读性"
                )
    
    def _check_line_length(self, file_path: Path, lines: List[str]) -> None:
        """检查行长度"""
        max_length = self.style_rules['max_line_length']
        
        for i, line in enumerate(lines, 1):
            if len(line) > max_length:
                self.add_issue(
                    str(file_path), i, "line_too_long", "warning",
                    f"行长度超过限制: {len(line)} > {max_length}",
                    "将长行拆分为多行"
                )
    
    def _check_imports(self, file_path: Path, tree: ast.AST) -> None:
        """检查导入语句"""
        imports = []
        import_lines = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
                import_lines.append(node.lineno)
        
        self.metrics.imports += len(imports)
        
        # 检查导入顺序
        if len(import_lines) > 1:
            # 简单检查：导入应该在文件开头
            first_non_import_line = None
            for node in ast.walk(tree):
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)) 
                    and hasattr(node, 'lineno')):
                    if first_non_import_line is None or node.lineno < first_non_import_line:
                        first_non_import_line = node.lineno
            
            if first_non_import_line:
                late_imports = [line for line in import_lines if line > first_non_import_line]
                if late_imports:
                    self.add_issue(
                        str(file_path), late_imports[0], "late_import", "warning",
                        "导入语句应该在文件开头",
                        "将所有导入移到文件顶部"
                    )
        
        # 检查未使用的导入（简单检查）
        self._check_unused_imports(file_path, tree, imports)
    
    def _check_unused_imports(self, file_path: Path, tree: ast.AST, imports: List[ast.AST]) -> None:
        """检查未使用的导入"""
        # 收集所有使用的名称
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # 检查导入的名称
        for import_node in imports:
            if isinstance(import_node, ast.Import):
                for alias in import_node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    if name not in used_names:
                        self.add_issue(
                            str(file_path), import_node.lineno, "unused_import", "info",
                            f"未使用的导入: {alias.name}",
                            "删除未使用的导入"
                        )
            elif isinstance(import_node, ast.ImportFrom):
                for alias in import_node.names:
                    if alias.name != '*':  # 忽略 from module import *
                        name = alias.asname if alias.asname else alias.name
                        if name not in used_names:
                            self.add_issue(
                                str(file_path), import_node.lineno, "unused_import", "info",
                                f"未使用的导入: {alias.name}",
                                "删除未使用的导入"
                            )
    
    def _check_functions(self, file_path: Path, tree: ast.AST) -> None:
        """检查函数"""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        self.metrics.functions += len(functions)
        
        max_length = self.style_rules['max_function_length']
        
        for func in functions:
            # 计算函数长度
            if hasattr(func, 'end_lineno') and func.end_lineno:
                func_length = func.end_lineno - func.lineno + 1
                if func_length > max_length:
                    self.add_issue(
                        str(file_path), func.lineno, "function_too_long", "warning",
                        f"函数 '{func.name}' 过长: {func_length} 行 > {max_length} 行",
                        "考虑将大函数拆分为更小的函数"
                    )
            
            # 检查参数数量
            arg_count = len(func.args.args)
            if arg_count > 5:
                self.add_issue(
                    str(file_path), func.lineno, "too_many_parameters", "warning",
                    f"函数 '{func.name}' 参数过多: {arg_count} 个",
                    "考虑使用配置对象或减少参数数量"
                )
    
    def _check_classes(self, file_path: Path, tree: ast.AST) -> None:
        """检查类"""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.metrics.classes += len(classes)
        
        max_length = self.style_rules['max_class_length']
        
        for cls in classes:
            # 计算类长度
            if hasattr(cls, 'end_lineno') and cls.end_lineno:
                class_length = cls.end_lineno - cls.lineno + 1
                if class_length > max_length:
                    self.add_issue(
                        str(file_path), cls.lineno, "class_too_long", "warning",
                        f"类 '{cls.name}' 过长: {class_length} 行 > {max_length} 行",
                        "考虑将大类拆分为更小的类"
                    )
            
            # 检查方法数量
            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
            if len(methods) > 20:
                self.add_issue(
                    str(file_path), cls.lineno, "too_many_methods", "warning",
                    f"类 '{cls.name}' 方法过多: {len(methods)} 个",
                    "考虑使用组合或继承来减少方法数量"
                )
    
    def _check_naming_conventions(self, file_path: Path, tree: ast.AST) -> None:
        """检查命名约定"""
        conventions = self.style_rules['naming_conventions']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(conventions['function'], node.name):
                    self.add_issue(
                        str(file_path), node.lineno, "naming_convention", "warning",
                        f"函数名 '{node.name}' 不符合命名约定",
                        "使用小写字母和下划线的函数名"
                    )
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(conventions['class'], node.name):
                    self.add_issue(
                        str(file_path), node.lineno, "naming_convention", "warning",
                        f"类名 '{node.name}' 不符合命名约定",
                        "使用驼峰命名法的类名"
                    )
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # 检查常量（全大写）
                        if name.isupper():
                            if not re.match(conventions['constant'], name):
                                self.add_issue(
                                    str(file_path), node.lineno, "naming_convention", "info",
                                    f"常量名 '{name}' 不符合命名约定",
                                    "使用全大写字母和下划线的常量名"
                                )
                        # 检查变量
                        else:
                            if not re.match(conventions['variable'], name):
                                self.add_issue(
                                    str(file_path), node.lineno, "naming_convention", "info",
                                    f"变量名 '{name}' 不符合命名约定",
                                    "使用小写字母和下划线的变量名"
                                )
    
    def _check_docstrings(self, file_path: Path, tree: ast.AST) -> None:
        """检查文档字符串"""
        if not self.style_rules['require_docstrings']:
            return
        
        # 检查模块文档字符串
        if not ast.get_docstring(tree):
            self.add_issue(
                str(file_path), 1, "missing_docstring", "warning",
                "缺少模块文档字符串",
                "在文件开头添加模块文档字符串"
            )
        
        # 检查函数和类的文档字符串
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
                else:
                    # 跳过私有方法和特殊方法的检查
                    if not (node.name.startswith('_') and not node.name.startswith('__')):
                        self.add_issue(
                            str(file_path), node.lineno, "missing_docstring", "info",
                            f"函数 '{node.name}' 缺少文档字符串",
                            "添加函数文档字符串说明参数、返回值和功能"
                        )
            
            elif isinstance(node, ast.ClassDef):
                total_classes += 1
                if ast.get_docstring(node):
                    documented_classes += 1
                else:
                    self.add_issue(
                        str(file_path), node.lineno, "missing_docstring", "warning",
                        f"类 '{node.name}' 缺少文档字符串",
                        "添加类文档字符串说明类的用途和功能"
                    )
        
        # 计算文档字符串覆盖率
        total_items = total_functions + total_classes
        documented_items = documented_functions + documented_classes
        if total_items > 0:
            coverage = documented_items / total_items
            self.metrics.docstring_coverage += coverage
    
    def _check_complexity(self, file_path: Path, tree: ast.AST) -> None:
        """检查圈复杂度"""
        max_complexity = self.style_rules['max_complexity']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                self.metrics.complexity += complexity
                
                if complexity > max_complexity:
                    self.add_issue(
                        str(file_path), node.lineno, "high_complexity", "warning",
                        f"函数 '{node.name}' 复杂度过高: {complexity} > {max_complexity}",
                        "简化函数逻辑或拆分为更小的函数"
                    )
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数的圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _check_code_smells(self, file_path: Path, tree: ast.AST, lines: List[str]) -> None:
        """检查代码异味"""
        # 检查重复代码（简单版本）
        self._check_duplicate_lines(file_path, lines)
        
        # 检查魔法数字
        self._check_magic_numbers(file_path, tree)
        
        # 检查空的异常处理
        self._check_empty_except(file_path, tree)
        
        # 检查过长的参数列表
        self._check_long_parameter_lists(file_path, tree)
    
    def _check_duplicate_lines(self, file_path: Path, lines: List[str]) -> None:
        """检查重复行"""
        line_counts = Counter(line.strip() for line in lines if line.strip())
        
        for line, count in line_counts.items():
            if count > 3 and len(line) > 20:  # 忽略短行和少量重复
                self.add_issue(
                    str(file_path), 0, "duplicate_code", "info",
                    f"发现重复代码行 ({count} 次): {line[:50]}...",
                    "考虑提取重复代码为函数或常量"
                )
    
    def _check_magic_numbers(self, file_path: Path, tree: ast.AST) -> None:
        """检查魔法数字"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                # 忽略常见的非魔法数字
                if node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                    self.add_issue(
                        str(file_path), node.lineno, "magic_number", "info",
                        f"魔法数字: {node.value}",
                        "将魔法数字定义为命名常量"
                    )
    
    def _check_empty_except(self, file_path: Path, tree: ast.AST) -> None:
        """检查空的异常处理"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    self.add_issue(
                        str(file_path), node.lineno, "empty_except", "warning",
                        "空的异常处理块",
                        "添加适当的异常处理逻辑或至少记录异常"
                    )
    
    def _check_long_parameter_lists(self, file_path: Path, tree: ast.AST) -> None:
        """检查过长的参数列表"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_args = (
                    len(node.args.args) + 
                    len(node.args.posonlyargs) + 
                    len(node.args.kwonlyargs) +
                    (1 if node.args.vararg else 0) +
                    (1 if node.args.kwarg else 0)
                )
                
                if total_args > 7:
                    self.add_issue(
                        str(file_path), node.lineno, "too_many_parameters", "warning",
                        f"函数 '{node.name}' 参数过多: {total_args} 个",
                        "考虑使用配置对象、数据类或减少参数数量"
                    )
    
    def check_project(self, project_path: Path = None) -> QualityReport:
        """检查整个项目"""
        if project_path is None:
            project_path = project_root
        
        self.logger.info(f"开始检查项目: {project_path}")
        start_time = time.time()
        
        # 遍历所有Python文件
        python_files = list(project_path.rglob("*.py"))
        self.logger.info(f"找到 {len(python_files)} 个Python文件")
        
        for file_path in python_files:
            # 跳过某些目录
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if '__pycache__' in str(file_path):
                continue
            
            self.check_file(file_path)
        
        # 计算平均文档字符串覆盖率
        if self.file_count > 0:
            self.metrics.docstring_coverage /= self.file_count
        
        # 生成报告
        duration = time.time() - start_time
        
        # 统计问题
        issues_by_severity = Counter(issue.severity for issue in self.issues)
        issues_by_type = Counter(issue.issue_type for issue in self.issues)
        
        # 生成建议
        recommendations = self._generate_recommendations()
        
        report = QualityReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_files=self.file_count,
            total_issues=len(self.issues),
            issues_by_severity=dict(issues_by_severity),
            issues_by_type=dict(issues_by_type),
            metrics=self.metrics,
            issues=self.issues,
            recommendations=recommendations
        )
        
        self.logger.info(f"代码质量检查完成，耗时: {duration:.2f}秒")
        self.logger.info(f"检查了 {self.file_count} 个文件，发现 {len(self.issues)} 个问题")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于问题类型生成建议
        issue_types = Counter(issue.issue_type for issue in self.issues)
        
        if issue_types.get('missing_docstring', 0) > 5:
            recommendations.append("增加文档字符串覆盖率，特别是公共API")
        
        if issue_types.get('line_too_long', 0) > 10:
            recommendations.append("配置代码格式化工具（如black）来自动处理行长度")
        
        if issue_types.get('high_complexity', 0) > 3:
            recommendations.append("重构复杂函数，使用更简单的逻辑结构")
        
        if issue_types.get('unused_import', 0) > 5:
            recommendations.append("使用工具（如autoflake）自动删除未使用的导入")
        
        if issue_types.get('naming_convention', 0) > 5:
            recommendations.append("统一命名约定，考虑使用pylint或flake8")
        
        # 基于指标生成建议
        if self.metrics.docstring_coverage < 0.5:
            recommendations.append("提高文档字符串覆盖率至少到50%")
        
        if self.metrics.comment_lines / max(self.metrics.code_lines, 1) < 0.1:
            recommendations.append("增加代码注释以提高可读性")
        
        # 通用建议
        recommendations.extend([
            "设置持续集成来自动运行代码质量检查",
            "使用pre-commit钩子在提交前检查代码质量",
            "定期进行代码审查以保持代码质量",
            "考虑使用类型提示来提高代码可读性和可维护性"
        ])
        
        return recommendations
    
    def save_report(self, report: QualityReport) -> None:
        """保存质量报告"""
        # 保存JSON报告
        json_file = self.output_dir / "code_quality_report.json"
        
        # 转换为可序列化的格式
        report_dict = {
            "timestamp": report.timestamp,
            "total_files": report.total_files,
            "total_issues": report.total_issues,
            "issues_by_severity": report.issues_by_severity,
            "issues_by_type": report.issues_by_type,
            "metrics": {
                "total_lines": report.metrics.total_lines,
                "code_lines": report.metrics.code_lines,
                "comment_lines": report.metrics.comment_lines,
                "blank_lines": report.metrics.blank_lines,
                "functions": report.metrics.functions,
                "classes": report.metrics.classes,
                "complexity": report.metrics.complexity,
                "imports": report.metrics.imports,
                "docstring_coverage": report.metrics.docstring_coverage
            },
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                }
                for issue in report.issues
            ],
            "recommendations": report.recommendations
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        # 生成可读的文本报告
        text_file = self.output_dir / "code_quality_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("FlorenceForge 代码质量报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"检查时间: {report.timestamp}\n")
            f.write(f"检查文件数: {report.total_files}\n")
            f.write(f"发现问题数: {report.total_issues}\n\n")
            
            # 代码指标
            f.write("代码指标:\n")
            f.write("-" * 20 + "\n")
            f.write(f"总行数: {report.metrics.total_lines}\n")
            f.write(f"代码行数: {report.metrics.code_lines}\n")
            f.write(f"注释行数: {report.metrics.comment_lines}\n")
            f.write(f"空白行数: {report.metrics.blank_lines}\n")
            f.write(f"函数数: {report.metrics.functions}\n")
            f.write(f"类数: {report.metrics.classes}\n")
            f.write(f"总复杂度: {report.metrics.complexity}\n")
            f.write(f"导入数: {report.metrics.imports}\n")
            f.write(f"文档字符串覆盖率: {report.metrics.docstring_coverage:.1%}\n\n")
            
            # 问题统计
            f.write("问题统计（按严重程度）:\n")
            f.write("-" * 30 + "\n")
            for severity, count in report.issues_by_severity.items():
                f.write(f"{severity}: {count}\n")
            f.write("\n")
            
            f.write("问题统计（按类型）:\n")
            f.write("-" * 30 + "\n")
            for issue_type, count in sorted(report.issues_by_type.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{issue_type}: {count}\n")
            f.write("\n")
            
            # 改进建议
            f.write("改进建议:\n")
            f.write("-" * 20 + "\n")
            for i, recommendation in enumerate(report.recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # 详细问题列表（只显示前50个）
            f.write("详细问题列表（前50个）:\n")
            f.write("-" * 30 + "\n")
            for issue in report.issues[:50]:
                f.write(f"[{issue.severity.upper()}] {issue.file_path}:{issue.line_number}\n")
                f.write(f"  类型: {issue.issue_type}\n")
                f.write(f"  消息: {issue.message}\n")
                if issue.suggestion:
                    f.write(f"  建议: {issue.suggestion}\n")
                f.write("\n")
        
        self.logger.info(f"质量报告已保存到: {json_file}")
        self.logger.info(f"可读报告已保存到: {text_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FlorenceForge代码质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
这个工具会分析FlorenceForge框架的代码质量，包括：
- 代码风格检查
- 命名约定检查
- 复杂度分析
- 文档字符串覆盖率
- 代码异味检测
- 导入分析
- 代码指标统计

示例用法:
  python code_quality_checker.py                    # 检查整个项目
  python code_quality_checker.py --output-dir ./quality_results  # 指定输出目录
        """
    )
    
    parser.add_argument(
        "--project-path", 
        type=Path,
        default=None,
        help="项目路径（默认为当前项目）"
    )
    parser.add_argument(
        "--output-dir", 
        default="./code_quality_results", 
        help="结果输出目录"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行代码质量检查
    checker = CodeQualityChecker(output_dir=args.output_dir)
    report = checker.check_project(args.project_path)
    
    # 保存报告
    checker.save_report(report)
    
    # 输出简要结果
    print("\n" + "="*60)
    print("FlorenceForge 代码质量检查结果")
    print("="*60)
    
    print(f"检查文件数: {report.total_files}")
    print(f"发现问题数: {report.total_issues}")
    print(f"代码行数: {report.metrics.code_lines}")
    print(f"函数数: {report.metrics.functions}")
    print(f"类数: {report.metrics.classes}")
    print(f"文档覆盖率: {report.metrics.docstring_coverage:.1%}")
    
    if report.issues_by_severity:
        print("\n问题分布:")
        for severity, count in report.issues_by_severity.items():
            print(f"  {severity}: {count}")
    
    print("\n主要建议:")
    for i, recommendation in enumerate(report.recommendations[:5], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n详细报告已保存到: {args.output_dir}")
    
    # 根据问题严重程度设置退出码
    error_count = report.issues_by_severity.get('error', 0)
    if error_count > 0:
        print(f"\n❌ 发现 {error_count} 个错误，请修复后重试。")
        sys.exit(1)
    else:
        print("\n✅ 代码质量检查完成。")
        sys.exit(0)

if __name__ == "__main__":
    main()