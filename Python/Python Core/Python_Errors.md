# Python Environment Debugging and Fixing Guide

## Problem Summary

While trying to install `jupyterlab` with `pip3`, the following error appeared:

```bash
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try brew install xyz
...
hint: See PEP 668 for the detailed specification.
````

At the same time, the shell configuration in `.zshrc` had multiple Python-related issues, including:

* `python` being aliased to `python2`
* invalid `PATH` entries pointing to binaries instead of directories
* conflicting Python environments from `pyenv`, Homebrew, and old Python 2
* duplicate and conflicting toolchain paths

---

## Root Causes

### 1. `pip3` was using Homebrew Python

Running these commands showed:

```bash
which python
which python3
python3 --version
pip3 --version
```

Output:

```bash
python: aliased to python2
/opt/homebrew/bin/python3
Python 3.14.2
pip 25.3 from /opt/homebrew/lib/python3.14/site-packages/pip (python 3.14)
```

This confirmed:

* `python3` was coming from Homebrew
* `pip3` was attached to Homebrew Python
* Homebrew Python blocks direct global installs via `pip3` because of PEP 668

That is why this failed:

```bash
pip3 install jupyterlab
```

---

### 2. `python` was incorrectly aliased to Python 2

The `.zshrc` included:

```bash
alias python=python2
```

This caused `python` to resolve to Python 2 instead of modern Python 3, which is not suitable for modern package installation or JupyterLab.

---

### 3. Some `PATH` entries were invalid

The original `.zshrc` had lines like:

```bash
export PATH="/Users/dev/.pyenv/versions/2.7.18/bin/python2:$PATH"
export PATH="/usr/local/bin/python3.8:$PATH"
```

These are incorrect because `PATH` should contain **directories**, not full executable file paths.

---

### 4. `pyenv` was still pointing `python` to an old broken Python 2.7

After removing the alias, `python` pointed to:

```bash
/Users/dev/.pyenv/shims/python
```

But running:

```bash
python --version
```

gave this error:

```bash
dyld: Library not loaded: /usr/local/opt/gettext/lib/libintl.8.dylib
Referenced from: /Users/dev/.pyenv/versions/2.7.18/bin/python2.7
Reason: tried: '/usr/local/opt/gettext/lib/libintl.8.dylib' (no such file)
zsh: abort python --version
```

This showed that `pyenv` was still resolving `python` to a broken Python 2.7.18 build linked against an old missing Homebrew library.

---

## Issues That Were Found

### Issue 1: `python` aliased to Python 2

**Problem:**

```bash
alias python=python2
```

**Why it was bad:**

* forced `python` to an obsolete interpreter
* broke modern Python workflows
* caused confusion between `python`, `python3`, and `pip3`

**Fix:**
Remove this line from `.zshrc`.

---

### Issue 2: Broken `pyenv` default version

**Problem:**
`python` was resolving to a broken Python 2.7.18 under pyenv.

**Why it was bad:**

* Python crashed before running
* shell looked correct at first, but actual interpreter was unusable

**Fix:**
Switch pyenv to a valid Python 3 version.

Final working result:

```bash
python --version
Python 3.12.10
```

---

### Issue 3: `pip3` tied to Homebrew Python

**Problem:**
`pip3` came from Homebrew Python 3.14.

**Why it was bad:**

* global `pip3 install ...` is blocked by PEP 668
* caused `externally-managed-environment` error

**Fix:**
Do not use `pip3 install ...` globally for this setup.
Use either:

* `python -m pip install ...`
* a virtual environment
* `pipx` for standalone tools

---

### Issue 4: Wrong `PATH` entries

**Problem examples:**

```bash
export PATH="/Users/dev/.pyenv/versions/2.7.18/bin/python2:$PATH"
export PATH="/usr/local/bin/python3.8:$PATH"
export PATH="/usr/local/opt/asdf/libexec/asdf.sh:$PATH"
```

**Why they were bad:**

* `PATH` should contain directories only
* `asdf.sh` is a script and should be sourced, not added to `PATH`

**Fixes:**

Correct form:

```bash
export PATH="/Users/dev/.pyenv/versions/2.7.18/bin:$PATH"
```

And for `asdf`:

```bash
[ -s "/usr/local/opt/asdf/libexec/asdf.sh" ] && . "/usr/local/opt/asdf/libexec/asdf.sh"
```

---

## Commands Used to Debug the Problem

### Step 1: Check which Python and pip are being used

```bash
which python
which python3
python3 --version
pip3 --version
```

Purpose:

* verify whether Python is coming from alias, pyenv, or Homebrew
* verify which interpreter `pip3` belongs to

---

### Step 2: Confirm whether `python` was aliased

```bash
which python
```

Output showed:

```bash
python: aliased to python2
```

That confirmed the alias issue.

---

### Step 3: Remove/fix alias and re-check

After cleanup:

```bash
which python
```

Output:

```bash
/Users/dev/.pyenv/shims/python
```

This confirmed alias removal and that `pyenv` was now controlling `python`.

---

### Step 4: Test the active `python`

```bash
python --version
```

This initially crashed because pyenv was still pointing to broken Python 2.7.18.

---

### Step 5: Fix pyenv version

Useful commands:

```bash
pyenv versions
pyenv version
pyenv global
```

Then switch to a valid Python 3 version:

```bash
pyenv global 3.12.10
pyenv rehash
```

Then verify:

```bash
python --version
```

Final result:

```bash
Python 3.12.10
```

---

## Commands That Fixed the Problem

### Fix 1: Remove Python 2 alias

Remove this from `.zshrc`:

```bash
alias python=python2
```

Optional legacy helper instead:

```bash
alias py2="/Users/dev/.pyenv/versions/2.7.18/bin/python2"
```

---

### Fix 2: Correct broken `PATH` entries

Remove invalid lines like:

```bash
export PATH="/Users/dev/.pyenv/versions/2.7.18/bin/python2:$PATH"
export PATH="/usr/local/bin/python3.8:$PATH"
```

Use correct directory path instead:

```bash
export PATH="/Users/dev/.pyenv/versions/2.7.18/bin:$PATH"
```

---

### Fix 3: Fix `asdf` setup

Remove:

```bash
export PATH="/usr/local/opt/asdf/libexec/asdf.sh:$PATH"
```

Use:

```bash
[ -s "/usr/local/opt/asdf/libexec/asdf.sh" ] && . "/usr/local/opt/asdf/libexec/asdf.sh"
```

---

### Fix 4: Refresh shell config

After editing `.zshrc`:

```bash
source ~/.zshrc
hash -r
```

---

### Fix 5: Set pyenv to a working Python 3 version

```bash
pyenv global 3.12.10
pyenv rehash
python --version
```

Working result:

```bash
Python 3.12.10
```

---

## Final Working Interpretation

At the end, the environment became:

* `python` → pyenv Python 3.12.10
* `python3` → Homebrew Python 3.14.2
* `pip3` → Homebrew pip 3.14

Because of that:

* `python -m pip` is the safer choice for package installation
* `pip3 install ...` should be avoided for global installs in this setup

---

## Correct Way to Install JupyterLab

### Recommended option: use a virtual environment

```bash
python -m venv ~/.venvs/jupyter
source ~/.venvs/jupyter/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab
jupyter lab
```

### Alternative: install with pipx

```bash
brew install pipx
pipx ensurepath
pipx install jupyterlab
```

---

## Commands That Should Be Avoided

Avoid:

```bash
pip3 install jupyterlab
```

Reason:

* `pip3` is tied to Homebrew Python
* Homebrew blocks global pip installs in that environment

Also avoid bringing back:

```bash
alias python=python2
```

---

## Recommended `.zshrc` Python-Related Block

```bash
# Python
export PATH="${HOME}/.pyenv/shims:${PATH}"
export PATH="/Users/dev/Library/Python/3.9/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# Optional legacy Python 2 helper
alias py2="/Users/dev/.pyenv/versions/2.7.18/bin/python2"
```

---

## Recommended General Cleanup Notes

The shell config also had multiple duplicate or conflicting paths for:

* PostgreSQL 13 / 15 / 16 / 17
* MySQL 8.0 / 8.4
* `libpq`
* OpenJDK
* PHP

These were not the direct cause of the JupyterLab issue, but they can create future command conflicts.

Best practice:

* keep only one active version per tool in `PATH`
* remove duplicates
* source scripts instead of adding them to `PATH`

---

## Useful Debug Commands for the Future

### Check active Python commands

```bash
which python
which python3
which pip
which pip3
python --version
python3 --version
python -m pip --version
python3 -m pip --version
```

### Check pyenv status

```bash
pyenv version
pyenv versions
pyenv global
```

### Reload shell

```bash
source ~/.zshrc
hash -r
```

---

## Final Result

The issue was solved by:

1. removing the `python=python2` alias
2. fixing invalid `PATH` lines
3. correcting `asdf` setup
4. switching pyenv to a working Python 3 version
5. using `python -m pip` or a virtual environment instead of global `pip3`

Final confirmation:

```bash
python --version
Python 3.12.10
```

---

## Recommended Safe Workflow Going Forward

For standalone tools like JupyterLab:

```bash
pipx install jupyterlab
```

For project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For general package installation:

```bash
python -m pip install <package-name>
```

Instead of:

```bash
pip install <package-name>
pip3 install <package-name>
```

```

If you want, I can also turn this into a cleaner GitHub-style `README.md` version with sections like **Symptoms**, **Investigation**, **Root Cause**, **Fix**, and **Prevention**.
```
