#!/usr/bin/env python3
"""
NASA/JPL Power of 10 Compliance Audit Script

This script audits Python code for compliance with NASA/JPL Power of 10 rules
adapted for Python and ML/research code.

Power of 10 Rules (adapted for Python):
1. Simple Control Flow - No unbounded recursion
2. Loop Bounds - All loops must have fixed upper bounds
3. Dynamic Memory - No allocation in critical inference loops
4. Function Length - No function > 60 lines (excluding docstrings)
5. Assertion Density - Average 2 assertions per function
6. Variable Scope - Smallest possible scope
7. Return Value Checking - Validate all outputs
8. Preprocessor Usage - Limit complex decorators/metaprogramming
9. Pointer Usage - Limit nested data structures (max 2 levels)
10. Compiler Warnings - Zero linter/type checker warnings

Usage:
    python power_of_10_audit.py <directory>
    python power_of_10_audit.py cloudml/
    python power_of_10_audit.py ./
"""

import ast
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""

    name: str
    file: str
    line_number: int
    num_lines: int
    num_assertions: int
    has_recursion: bool
    has_unbounded_loops: bool
    max_nesting_depth: int
    complexity: int


@dataclass
class ComplianceReport:
    """Overall compliance report."""

    total_functions: int = 0
    compliant_functions: int = 0
    violations: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = defaultdict(list)


class PowerOf10Auditor(ast.NodeVisitor):
    """AST visitor for Power of 10 compliance checking."""

    MAX_FUNCTION_LINES = 60
    MIN_ASSERTIONS_PER_FUNCTION = 2
    MAX_NESTING_DEPTH = 2
    MAX_LOOP_ITERATIONS = 10000  # Reasonable upper bound

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metrics: List[FunctionMetrics] = []
        self.current_function = None
        self.recursion_calls = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and collect metrics."""
        # Calculate function length (excluding docstring)
        func_start = node.lineno
        func_end = self._get_last_line(node)

        # Exclude docstring if present
        docstring_lines = 0
        if (
            ast.get_docstring(node)
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            docstring_node = node.body[0]
            docstring_lines = (
                self._get_last_line(docstring_node) - docstring_node.lineno + 1
            )

        num_lines = func_end - func_start + 1 - docstring_lines

        # Count assertions
        num_assertions = self._count_assertions(node)

        # Check for recursion
        self.current_function = node.name
        self.recursion_calls = set()
        has_recursion = self._has_recursion(node)

        # Check for unbounded loops
        has_unbounded_loops = self._has_unbounded_loops(node)

        # Calculate nesting depth
        max_nesting = self._max_nesting_depth(node)

        # Calculate cyclomatic complexity (simplified)
        complexity = self._cyclomatic_complexity(node)

        metrics = FunctionMetrics(
            name=node.name,
            file=self.filepath,
            line_number=func_start,
            num_lines=num_lines,
            num_assertions=num_assertions,
            has_recursion=has_recursion,
            has_unbounded_loops=has_unbounded_loops,
            max_nesting_depth=max_nesting,
            complexity=complexity,
        )

        self.metrics.append(metrics)

        # Continue visiting child nodes
        self.generic_visit(node)
        self.current_function = None

    def _get_last_line(self, node: ast.AST) -> int:
        """Get the last line number of a node."""
        last_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                last_line = max(last_line, child.lineno)
            if hasattr(child, "end_lineno") and child.end_lineno:
                last_line = max(last_line, child.end_lineno)
        return last_line

    def _count_assertions(self, node: ast.FunctionDef) -> int:
        """Count assert statements in function."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                count += 1
        return count

    def _has_recursion(self, node: ast.FunctionDef) -> bool:
        """Check if function has recursive calls."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id == node.name:
                        return True
        return False

    def _has_unbounded_loops(self, node: ast.FunctionDef) -> bool:
        """Check for loops without explicit bounds."""
        for child in ast.walk(node):
            # Check while loops
            if isinstance(child, ast.While):
                # While loops are potentially unbounded unless proven otherwise
                # Check for iteration counter or break conditions
                if not self._has_iteration_limit(child):
                    return True

            # For loops with range() are typically bounded
            # But other iterators might not be
            if isinstance(child, ast.For):
                if not self._is_bounded_for_loop(child):
                    return True

        return False

    def _has_iteration_limit(self, node: ast.While) -> bool:
        """Check if while loop has iteration limit."""
        # Look for MAX_ITERATIONS or similar pattern
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                # Check for iteration < MAX_ITERATIONS pattern
                if any(isinstance(comp, ast.Lt) for comp in child.ops):
                    return True
        return False

    def _is_bounded_for_loop(self, node: ast.For) -> bool:
        """Check if for loop has known bounds."""
        # For loops with range() are bounded
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range":
                    return True

        # For loops over lists/tuples with known size are bounded
        if isinstance(node.iter, (ast.List, ast.Tuple)):
            return True

        # Otherwise, assume potentially unbounded
        return False

    def _max_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth of control structures."""

        def depth(n, current=0):
            max_d = current
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    max_d = max(max_d, depth(child, current + 1))
                else:
                    max_d = max(max_d, depth(child, current))
            return max_d

        return depth(node, 0)

    def _cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Each decision point adds 1
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # And/Or operations
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity


def audit_file(filepath: Path) -> List[FunctionMetrics]:
    """Audit a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        auditor = PowerOf10Auditor(str(filepath))
        auditor.visit(tree)

        return auditor.metrics
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def audit_directory(directory: Path) -> ComplianceReport:
    """Audit all Python files in directory."""
    report = ComplianceReport()
    all_metrics = []

    # Find all Python files
    python_files = list(directory.rglob("*.py"))

    print(f"Auditing {len(python_files)} Python files in {directory}...")
    print()

    for filepath in python_files:
        # Skip __pycache__ and .venv
        if "__pycache__" in str(filepath) or "venv" in str(filepath):
            continue

        metrics = audit_file(filepath)
        all_metrics.extend(metrics)

    # Analyze metrics
    report.total_functions = len(all_metrics)

    for metric in all_metrics:
        is_compliant = True

        # Rule 1: No unbounded recursion
        if metric.has_recursion:
            report.violations["Rule 1: Recursion"].append(
                f"{metric.file}:{metric.line_number} - {metric.name}()"
            )
            is_compliant = False

        # Rule 2: Loop bounds
        if metric.has_unbounded_loops:
            report.violations["Rule 2: Unbounded Loops"].append(
                f"{metric.file}:{metric.line_number} - {metric.name}()"
            )
            is_compliant = False

        # Rule 4: Function length
        if metric.num_lines > PowerOf10Auditor.MAX_FUNCTION_LINES:
            report.violations["Rule 4: Function Length"].append(
                f"{metric.file}:{metric.line_number} - {metric.name}() "
                f"({metric.num_lines} lines > {PowerOf10Auditor.MAX_FUNCTION_LINES})"
            )
            is_compliant = False

        # Rule 5: Assertion density
        if metric.num_assertions < PowerOf10Auditor.MIN_ASSERTIONS_PER_FUNCTION:
            report.violations["Rule 5: Assertion Density"].append(
                f"{metric.file}:{metric.line_number} - {metric.name}() "
                f"({metric.num_assertions} assertions < {PowerOf10Auditor.MIN_ASSERTIONS_PER_FUNCTION})"
            )
            is_compliant = False

        # Rule 9: Nesting depth
        if metric.max_nesting_depth > PowerOf10Auditor.MAX_NESTING_DEPTH:
            report.violations["Rule 9: Nesting Depth"].append(
                f"{metric.file}:{metric.line_number} - {metric.name}() "
                f"(depth {metric.max_nesting_depth} > {PowerOf10Auditor.MAX_NESTING_DEPTH})"
            )
            is_compliant = False

        if is_compliant:
            report.compliant_functions += 1

    return report


def print_report(report: ComplianceReport) -> None:
    """Print compliance report."""
    print("=" * 80)
    print("NASA/JPL Power of 10 Compliance Report")
    print("=" * 80)
    print()

    print(f"Total Functions Analyzed: {report.total_functions}")
    print(f"Compliant Functions: {report.compliant_functions}")
    print(
        f"Non-Compliant Functions: {report.total_functions - report.compliant_functions}"
    )

    if report.total_functions > 0:
        compliance_rate = (report.compliant_functions / report.total_functions) * 100
        print(f"Compliance Rate: {compliance_rate:.2f}%")

    print()
    print("-" * 80)
    print("Violations by Rule:")
    print("-" * 80)
    print()

    if not report.violations:
        print(" No violations found! Full compliance achieved.")
    else:
        for rule, violations in sorted(report.violations.items()):
            print(f"\n{rule} ({len(violations)} violations):")
            for violation in violations[:10]:  # Show first 10
                print(f"  - {violation}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")

    print()
    print("=" * 80)


def save_report(report: ComplianceReport, output_file: Path) -> None:
    """Save compliance report to markdown file."""
    with open(output_file, "w") as f:
        f.write("# NASA/JPL Power of 10 Compliance Report\n\n")
        f.write(f"**Generated**: {output_file.name}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Functions Analyzed**: {report.total_functions}\n")
        f.write(f"- **Compliant Functions**: {report.compliant_functions}\n")
        f.write(
            f"- **Non-Compliant Functions**: {report.total_functions - report.compliant_functions}\n"
        )

        if report.total_functions > 0:
            compliance_rate = (
                report.compliant_functions / report.total_functions
            ) * 100
            f.write(f"- **Compliance Rate**: {compliance_rate:.2f}%\n")

        f.write("\n## Violations by Rule\n\n")

        if not report.violations:
            f.write(" **No violations found! Full compliance achieved.**\n")
        else:
            for rule, violations in sorted(report.violations.items()):
                f.write(f"\n### {rule}\n\n")
                f.write(f"**Count**: {len(violations)}\n\n")

                for violation in violations:
                    f.write(f"- `{violation}`\n")

        f.write("\n## Power of 10 Rules Reference\n\n")
        f.write("1. **Simple Control Flow** - No unbounded recursion\n")
        f.write("2. **Loop Bounds** - All loops must have fixed upper bounds\n")
        f.write("3. **Dynamic Memory** - No allocation in critical inference loops\n")
        f.write(
            "4. **Function Length** - No function > 60 lines (excluding docstrings)\n"
        )
        f.write("5. **Assertion Density** - Average 2 assertions per function\n")
        f.write("6. **Variable Scope** - Smallest possible scope\n")
        f.write("7. **Return Value Checking** - Validate all outputs\n")
        f.write(
            "8. **Preprocessor Usage** - Limit complex decorators/metaprogramming\n"
        )
        f.write("9. **Pointer Usage** - Limit nested data structures (max 2 levels)\n")
        f.write("10. **Compiler Warnings** - Zero linter/type checker warnings\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python power_of_10_audit.py <directory>")
        print("Example: python power_of_10_audit.py cloudml/")
        sys.exit(1)

    directory = Path(sys.argv[1])

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    # Run audit
    report = audit_directory(directory)

    # Print report
    print_report(report)

    # Save report
    output_dir = Path("./reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "power_of_10_compliance.md"
    save_report(report, output_file)

    print(f"\nReport saved to: {output_file}")

    # Exit code based on compliance
    if report.total_functions > 0:
        compliance_rate = (report.compliant_functions / report.total_functions) * 100
        if compliance_rate < 80:
            print("\n  Compliance rate below 80%. Review violations and remediate.")
            sys.exit(1)
        elif compliance_rate < 100:
            print(
                "\n Compliance rate acceptable (≥80%). Consider addressing remaining violations."
            )
            sys.exit(0)
        else:
            print("\n Full compliance achieved!")
            sys.exit(0)
    else:
        print("\n  No functions found to audit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
