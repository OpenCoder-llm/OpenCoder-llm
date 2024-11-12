"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import sys
import pathlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import ast
import traceback

from typing import Dict, List, Optional, Set, Tuple

from utils import refine_text, python_extract


def syntax_check(code, verbose = False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()

    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""

    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i:j+1])
            if syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i:j+1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet

    return max_valid_snippet

def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps

def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]

    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)

    return visited

def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None

def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))

def sanitize(text: str, entrypoint: Optional[str] = None) -> str:

    text = refine_text(text)

    # text = python_extract(text)

    code = extract_longest_valid_code(text)
    tree = ast.parse(code)
    
    definitions = {}

    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            definitions[name] = ('class', node)
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            if has_return_statement(node):
                definitions[name] = ('function', node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ('variable', node)

    if entrypoint:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = []

    for node in imports:
        sanitized_output.append(ast.unparse(node))

    for name, (_, node) in definitions.items():
        if not entrypoint or name in reachable:
            sanitized_output.append(ast.unparse(node))

    return "\n".join(sanitized_output)
