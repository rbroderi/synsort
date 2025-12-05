# synsort

`synsort` is a structural linter that keeps module level globals, classes, functions, and methods grouped and ordered. It is designed for teams that want deterministic symbol layout without hand-editing large files.

## Highlights

- Creates four labeled sections (Globals, Dunder Members, Public Members, Private Members) for every sortable cluster.
- Sorts within each section by usage frequency (most referenced names stay closest to the top) with deterministic fallbacks.
- Reparents contiguous comment blocks so they follow the definition they describe.
- Recursively reorders class methods using the same rules as module level code.
- Uses a JSON cache (`.synsort_cache.json`) so untouched files are skipped quickly.

## Installation & Usage

```powershell
pip install -e .
synsort src/some_module.py
```

Useful flags:

- `synsort --check path/` – report files that would change and exit with code 1.
- `synsort --no-cache` – bypass the cache layer.
- `synsort --clear-cache` – drop cached hashes before sorting.

You can also run it as a module: `python -m synsort path/to/file.py`.

## Configuration

Configuration lives under `[tool.synsort]` inside `pyproject.toml`.

```toml
[tool.synsort]
order = ["globals", "dunder", "public", "private"]
cache_file = ".synsort_cache.json"

[tool.synsort.section_headers]
globals = "# === Module Globals ==="
public = "# === Public API ==="
```

- `order` – choose the category ordering. Missing categories are appended automatically.
- `cache_file` – optionally relocate the cache inside your repo.
- `section_headers` – override the banner text per category.

## How ordering works

1. The file is parsed with `ast` to capture every top-level assignment, function, and class. Nested functions are intentionally skipped.
2. Every identifier usage (`ast.Name` loads and attribute names) is counted, and items are sorted by `usage desc → name asc → original order` inside their category.
3. The four categories are emitted in the configured order with headers preceding each one.
4. For classes, the same pipeline is rerun against their methods while keeping docstrings, attributes, and other statements in place.

## Cache behavior

`synsort` tracks a SHA-256 digest plus the active configuration signature per file. When the digest and config both match the cached entry the run is skipped. Change any of the source text, ordering, or header labels and the next run will rewrite the file and refresh the cache. Cache writes are flushed at the end of every CLI invocation, and you can delete the entire file safely if you ever need a clean slate.

## Limitations

- Only `.py` files are considered, and nested function definitions are ignored for now.
- Method reordering occurs only inside stretches of methods separated by other statements (class attributes act as natural separators).
- Files with syntax errors are left untouched.

Feedback and improvements are welcome!
