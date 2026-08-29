import ast
import tomllib
from graphlib import TopologicalSorter
from pathlib import Path

import nbformat


def get_pyproject_reqs(path: str):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    return deps


def get_deps(path: Path, local_modules: set[str]) -> set[str]:
    deps = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception as e:
        print(f"\033[93mwarning:\033[0m file {path} could not be parsed\n{e}")
        return deps

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".")[0]
                if root_name in local_modules and root_name != path.stem:
                    deps.add(root_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_name = node.module.split(".")[0]
                if root_name in local_modules and root_name != path.stem:
                    deps.add(root_name)

    return deps


def remove_problems(contents: str, local_modules: set[str], remove__main__=True) -> str:
    tree = ast.parse(contents)

    lines = contents.splitlines()
    lines_pending_removal = []

    for node in ast.walk(tree):
        is_local = False
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in local_modules for alias in node.names):
                is_local = True
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or (
                node.module and node.module.split(".")[0] in local_modules
            ):
                is_local = True
        else:
            continue

        if is_local:
            for lineno in range(
                node.lineno,
                (node.end_lineno if node.end_lineno is not None else node.lineno) + 1,
            ):
                lines_pending_removal.append(lineno)

    lines = [line for i, line in enumerate(lines, 1) if i not in lines_pending_removal]
    contents = "\n".join(lines).strip()

    if remove__main__:
        contents = contents.replace("__main__", "__main__DISABLED")

    return contents


def build_notebook(
    requirements: list[str], pre_file: str, in_files: list[str], out_file: str
):
    in_paths = [Path(path) for path in in_files]
    out_path = Path(out_file)

    filemap = {file.stem: file for file in in_paths}
    local_modules = set(filemap.keys())

    graph = {stem: get_deps(path, local_modules) for stem, path in filemap.items()}

    sorter = TopologicalSorter(graph)
    try:
        dep_ordered_stems = list(sorter.static_order())
    except Exception as e:
        print(f"\033[91merror:\033[0m unable to sort files\n{e}")
        return

    nb = nbformat.v4.new_notebook()

    if pre_file:
        contents = Path(pre_file).read_text()
        pre_cell = nbformat.v4.new_code_cell(contents)
        nb.cells.append(pre_cell)

    if requirements:
        reqs = [f'"{req}"' for req in requirements]
        contents = f"%pip install {' '.join(reqs)}"
        pip_cell = nbformat.v4.new_code_cell(contents)
        nb.cells.append(pip_cell)

    cells = []
    for stem in dep_ordered_stems:
        path = filemap[stem]
        contents = path.read_text()
        contents = remove_problems(
            contents, local_modules, remove__main__=path.name != "main.py"
        )
        contents = f"# file: {path.name}\n\n" + contents
        cell = nbformat.v4.new_code_cell(contents)
        cell.metadata["filename"] = path.name
        nb.cells.append(cell)

    with out_path.open("w") as f:
        nbformat.write(nb, f)

    print(f"wrote notebook to {out_file}")


requirements = get_pyproject_reqs("pyproject.toml")
files = [
    "corpus.py",
    "tokenizer.py",
    "dataset.py",
    "embedding.py",
    "nca.py",
    "main.py",
]
build_notebook(
    requirements=[],
    pre_file="prebuild.py",
    in_files=files,
    out_file="build.ipynb",
)
