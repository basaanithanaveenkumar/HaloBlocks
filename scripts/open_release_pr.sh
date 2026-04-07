#!/usr/bin/env bash
# scripts/open_release_pr.sh — Bump pyproject.toml version, push branch release/vX.Y.Z, open PR.
# After the PR is merged to main, CI creates tag vX.Y.Z and the existing publish workflow ships to PyPI.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.2.0"
  exit 1
fi

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
  echo "Version must look like MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-rc1"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v gh &>/dev/null; then
  echo "Install GitHub CLI: https://cli.github.com/"
  exit 1
fi

if ! gh auth status &>/dev/null; then
  echo "Run: gh auth login"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash changes first."
  exit 1
fi

BRANCH="release/v${VERSION}"

git fetch origin main 2>/dev/null || true
git checkout main
git pull origin main
git checkout -b "$BRANCH"

export HALO_RELEASE_VERSION="$VERSION"
python3 <<'PY'
import os
import pathlib
import re
import sys

ver = os.environ["HALO_RELEASE_VERSION"]
path = pathlib.Path("pyproject.toml")
text = path.read_text()
new, n = re.subn(
    r"(?m)^version = \"[^\"]+\"",
    f'version = "{ver}"',
    text,
    count=1,
)
if n != 1:
    print("Could not find a single version = line in pyproject.toml", file=sys.stderr)
    sys.exit(1)
path.write_text(new)
PY

command -v uv &>/dev/null && uv lock || true

echo "Running formatter..."
bash "$ROOT_DIR/scripts/format.sh"

# --- Generate CHANGELOG entry ---------------------------------------------------
CHANGELOG="$ROOT_DIR/CHANGELOG.md"
TODAY=$(date +%Y-%m-%d)

PREV_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [[ -n "$PREV_TAG" ]]; then
  LOG_RANGE="${PREV_TAG}..HEAD"
  COMPARE_NOTE="**Full diff:** [\`${PREV_TAG}...v${VERSION}\`](../../compare/${PREV_TAG}...v${VERSION})"
else
  LOG_RANGE="HEAD"
  COMPARE_NOTE=""
fi

ENTRIES=$(git log "$LOG_RANGE" --pretty=format:"- %s (%h)" --no-merges)
if [[ -z "$ENTRIES" ]]; then
  ENTRIES="- Version bump to ${VERSION}"
fi

NEW_SECTION="## [v${VERSION}] - ${TODAY}

${ENTRIES}
${COMPARE_NOTE:+
$COMPARE_NOTE}"

python3 - "$CHANGELOG" "$NEW_SECTION" <<'PYLOG'
import pathlib, sys
changelog = pathlib.Path(sys.argv[1])
section = sys.argv[2]
text = changelog.read_text() if changelog.exists() else "# Changelog\n"
header_end = text.find("\n## ")
if header_end == -1:
    header_end = len(text)
text = text[:header_end].rstrip() + "\n\n" + section.strip() + "\n" + text[header_end:]
changelog.write_text(text)
PYLOG

echo "Updated CHANGELOG.md with v${VERSION} entry."
# ---------------------------------------------------------------------------------

git add pyproject.toml CHANGELOG.md
if [[ -f uv.lock ]]; then git add uv.lock; fi
git add -u src/ tests/
git commit -m "chore: release v${VERSION}"

git push -u origin "$BRANCH"

PR_BODY="## Release v${VERSION}

### Changes
${ENTRIES}

---
After this PR is **merged** to \`main\`, CI will create and push tag \`v${VERSION}\`, which triggers the PyPI publish workflow."

gh pr create \
  --base main \
  --head "$BRANCH" \
  --title "Release v${VERSION}" \
  --body "$PR_BODY"

echo "Opened PR for v${VERSION}. After merge, tagging and PyPI publish run automatically."
