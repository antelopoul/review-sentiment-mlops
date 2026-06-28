#!/usr/bin/env bash
# ============================================================
# build.sh — build, tag, and version-bump the sentiment API image
#
# Usage:
#   ./build.sh            # build current version, then bump patch
#   ./build.sh --no-bump  # build only, leave VERSION unchanged
#   ./build.sh --push     # build + push to registry (set REGISTRY below)
# ============================================================

set -euo pipefail

IMAGE_NAME="sentiment-api"
REGISTRY=""   # e.g. "gcr.io/my-project" — leave empty for local-only builds

# ---------------------------------------------------------------------------
# 0. Generate serving requirements (CPU-only, no torch, no dev tools)
# ---------------------------------------------------------------------------
# uv export --no-dev produces only [project].dependencies — everything in
# [dependency-groups] (train-cpu, train-gpu, dev) is excluded, so CUDA torch
# never lands in the image.
echo "Generating requirements-serve.txt (prod deps only, no CUDA) ..."
uv export --no-dev --format requirements-txt -o requirements-serve.txt
echo "✓ requirements-serve.txt generated ($(wc -l < requirements-serve.txt) packages)"

# ---------------------------------------------------------------------------
# 1. Resolve version + git SHA
# ---------------------------------------------------------------------------

SEMVER=$(cat VERSION | tr -d '[:space:]')
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
FULL_VERSION="${SEMVER}-${GIT_SHA}"   # e.g. "1.2.3-a4f9e1b"

echo "============================================"
echo "  Image    : ${IMAGE_NAME}"
echo "  Version  : ${SEMVER}"
echo "  Git SHA  : ${GIT_SHA}"
echo "  Tag      : ${FULL_VERSION}"
echo "============================================"

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------

TAGS=(
    "-t" "${IMAGE_NAME}:${SEMVER}"
    "-t" "${IMAGE_NAME}:${FULL_VERSION}"
    "-t" "${IMAGE_NAME}:latest"
)

if [[ -n "${REGISTRY}" ]]; then
    TAGS+=(
        "-t" "${REGISTRY}/${IMAGE_NAME}:${SEMVER}"
        "-t" "${REGISTRY}/${IMAGE_NAME}:${FULL_VERSION}"
        "-t" "${REGISTRY}/${IMAGE_NAME}:latest"
    )
fi

docker build \
    --build-arg "BUILD_VERSION=${FULL_VERSION}" \
    "${TAGS[@]}" \
    .

echo ""
echo "✓ Built:"
echo "  ${IMAGE_NAME}:${SEMVER}"
echo "  ${IMAGE_NAME}:${FULL_VERSION}"
echo "  ${IMAGE_NAME}:latest"

# ---------------------------------------------------------------------------
# 3. Push (optional — pass --push flag)
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--push" ]]; then
    if [[ -z "${REGISTRY}" ]]; then
        echo "ERROR: Set REGISTRY at the top of build.sh before pushing."
        exit 1
    fi
    docker push "${REGISTRY}/${IMAGE_NAME}:${SEMVER}"
    docker push "${REGISTRY}/${IMAGE_NAME}:${FULL_VERSION}"
    docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    echo "✓ Pushed to ${REGISTRY}"
fi

# ---------------------------------------------------------------------------
# 4. Bump patch version (unless --no-bump)
# ---------------------------------------------------------------------------

if [[ "${1:-}" != "--no-bump" ]]; then
    IFS='.' read -r MAJOR MINOR PATCH <<< "${SEMVER}"
    NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
    echo "${NEW_VERSION}" > VERSION
    echo ""
    echo "✓ VERSION bumped: ${SEMVER} → ${NEW_VERSION}"
    echo "  Commit VERSION before the next build:"
    echo "  git add VERSION && git commit -m \"chore: bump version to ${NEW_VERSION}\""
fi
