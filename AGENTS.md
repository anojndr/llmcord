Rules:

1. Always use `uv` for Python rather than any other method.
2. Always run `ruff check --select ALL . --fix` after making changes. Resolve all errors until none remain, and save the Ruff output to `.\tmp` so it is easy to review, especially if there are many errors.
3. Always run `ruff format` after making changes.
4. Always run `ty check` after making changes, then fix all errors.
5. Always run the tests after making changes, and fix any errors that appear.
6. Always use `ripgrep` instead of `grep`.
7. Always use `fd` instead of `find`.