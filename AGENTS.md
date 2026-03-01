rules:

1. Always use `uv` rather than any other method.
2. Make sure to respect the `.gitignore` file.
3. Always run `ruff check --select ALL . --fix` after making changes. Resolve all errors until none remain, and save the Ruff output to `.\tmp` so you can easily review the errors, especially if there are many.
4. Always run `ruff format` after making changes.
5. Always run `ty check` after making changes, and then fix all errors.
6. Always run `.\tests` after making changes and fix any errors that appear.
7. Always use ripgrep instead of grep.
8. Always use fd instead of find.
