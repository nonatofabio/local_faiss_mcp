# Publishing to PyPI

This guide explains how to publish the `local-faiss-mcp` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/
2. **Trusted Publishing**: Set up trusted publishing on PyPI (recommended) or use API tokens

## Option 1: Automated Publishing via GitHub Actions (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/publish.yml`) that automatically publishes to PyPI when you create a GitHub release.

### Setup Trusted Publishing (One-time)

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name**: `local-faiss-mcp`
   - **Owner**: `nonatofabio` (your GitHub username)
   - **Repository name**: `local_faiss_mcp`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave blank)
4. Click "Add"

### Publishing a New Version

1. **Update the version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment as needed
   ```

2. **Commit and push** your changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

3. **Create a GitHub release**:
   - Go to https://github.com/nonatofabio/local_faiss_mcp/releases/new
   - Create a new tag (e.g., `v0.1.1`)
   - Set the release title (e.g., `v0.1.1`)
   - Add release notes
   - Click "Publish release"

4. The GitHub Action will automatically:
   - Run tests
   - Build the package
   - Publish to PyPI

## Option 2: Manual Publishing

If you prefer to publish manually or need to troubleshoot:

1. **Ensure you're in the project directory with the virtual environment activated**:
   ```bash
   cd local_faiss_mcp
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Clean previous builds** (if any):
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

3. **Update version** in `pyproject.toml` if needed

4. **Build the package**:
   ```bash
   python -m build
   ```

5. **Check the package** (optional but recommended):
   ```bash
   twine check dist/*
   ```

6. **Upload to TestPyPI** (optional, for testing):
   ```bash
   twine upload --repository testpypi dist/*
   ```
   You'll need to create an account at https://test.pypi.org/ and generate an API token.

7. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```
   Enter your PyPI username and password/API token when prompted.

### Creating a PyPI API Token (for manual uploads)

1. Log in to https://pypi.org/
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name and select the scope
5. Copy the token (starts with `pypi-`)
6. Use it as your password when running `twine upload`

You can also configure credentials in `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
```

## Versioning

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): Add functionality in a backward compatible manner
- **PATCH** version (0.0.1): Backward compatible bug fixes

## Verification

After publishing, verify your package:

1. Check it appears on PyPI: https://pypi.org/project/local-faiss-mcp/
2. Install it in a clean environment:
   ```bash
   pip install local-faiss-mcp
   ```
3. Test that it works:
   ```bash
   python -c "import server; print('Success!')"
   ```

## Troubleshooting

### "File already exists" error
- You cannot replace a version once uploaded to PyPI
- Increment the version number and try again

### Package structure issues
- Ensure `pyproject.toml` has the correct `py-modules` setting
- Verify `MANIFEST.in` includes all necessary files
- Run `python -m build` locally first to catch issues

### Authentication errors
- For GitHub Actions: Ensure trusted publishing is configured correctly
- For manual upload: Verify your API token is correct and has the right permissions

## Resources

- PyPI Documentation: https://packaging.python.org/
- Trusted Publishing Guide: https://docs.pypi.org/trusted-publishers/
- Twine Documentation: https://twine.readthedocs.io/
