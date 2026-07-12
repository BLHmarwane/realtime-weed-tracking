"""Repository-level smoke contracts with no heavyweight runtime imports."""

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_REQUIREMENTS = [
    "ultralytics>=8.4,<9",
    "lap>=0.5,<1",
    "opencv-python>=4.10,<5",
    "pyyaml>=6,<7",
    "streamlit>=1.59,<2",
    "imageio-ffmpeg>=0.6,<1",
]
DEV_REQUIREMENTS = ["pytest>=8,<9", "pyyaml>=6,<7"]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _tracked_public_text_paths() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "*.py",
            "*.yaml",
            "*.yml",
            "docker/Dockerfile",
            "models/README.md",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        Path(relative_path)
        for relative_path in result.stdout.split("\0")
        if relative_path
        and (PROJECT_ROOT / relative_path).is_file()
    ]


def test_public_scan_includes_its_own_contract():
    assert Path("tests/test_public_contract.py") in _tracked_public_text_paths()


def test_runtime_requirements_are_exact_and_bounded():
    requirements = _read("requirements.txt").splitlines()

    assert requirements == RUNTIME_REQUIREMENTS
    for requirement in requirements:
        assert ">=" in requirement and "<" in requirement


def test_dev_requirements_are_exact_and_bounded():
    requirements = _read("requirements-dev.txt").splitlines()

    assert requirements == DEV_REQUIREMENTS
    for requirement in requirements:
        assert ">=" in requirement and "<" in requirement


def test_smoke_workflow_uses_the_lightweight_public_toolchain():
    workflow_path = PROJECT_ROOT / ".github/workflows/smoke-tests.yml"

    assert workflow_path.exists()
    assert not (PROJECT_ROOT / ".github/workflows/tests.yml").exists()

    workflow = workflow_path.read_text(encoding="utf-8")
    for expected in (
        "name: smoke-tests",
        "permissions:\n  contents: read",
        "actions/checkout@v7",
        "actions/setup-python@v6",
        'python-version: "3.12"',
        "cache: pip",
        "cache-dependency-path: requirements-dev.txt",
        "pip install -r requirements-dev.txt",
        "python -m pytest -q",
        "python -m compileall -q app weedtrack scripts",
    ):
        assert expected in workflow


def test_config_keeps_only_active_public_settings():
    config = _read("configs/config.yaml")

    assert "iou_threshold" not in config
    assert "max_upload_mb: 200" in config


def test_container_runs_as_a_dedicated_non_root_user():
    dockerfile = _read("docker/Dockerfile")

    for expected in (
        "HEALTHCHECK",
        "useradd --create-home --uid 10001 appuser",
        "mkdir -p /tmp/ultralytics /tmp/.streamlit",
        "chown -R appuser:appuser /tmp/ultralytics /tmp/.streamlit /app",
        "ENV HOME=/tmp",
        "USER 10001:10001",
    ):
        assert expected in dockerfile
    last_copy_position = dockerfile.index(
        "COPY data/sample_video.mp4 data/sample_video.mp4"
    )
    user_creation_position = dockerfile.index(
        "useradd --create-home --uid 10001 appuser"
    )
    assert last_copy_position < user_creation_position


def test_container_bounds_the_cpu_torch_stack():
    dockerfile = _read("docker/Dockerfile")
    cpu_install = dockerfile[
        dockerfile.index("RUN pip install --no-cache-dir") : dockerfile.index(
            "&& pip install --no-cache-dir -r requirements.txt"
        )
    ]

    assert '"torch>=2.12,<3"' in cpu_install
    assert '"torchvision>=0.27,<1"' in cpu_install


def test_public_sources_do_not_expose_private_or_completed_project_copy():
    private_document_refs = ("STATE" + ".md", "TUTORIAL" + ".md")
    milestone_pattern = re.compile(r"\bM[235]\b")
    unsupported_claims = (
        "identifiant" + " stable",
        "stable" + " identifier",
        "stable" + " identity",
        "ne pas traiter" + " deux fois",
        "never treats" + " the same weed twice",
        "guarantees treatment " + "deduplication",
        "guaranteed treatment " + "deduplication",
        "ils seront attachés à une " + "release",
        "will be attached to a future " + "release",
        "one-command published " + "image",
    )
    violations: list[str] = []

    for relative_path in _tracked_public_text_paths():
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        lowered = text.casefold()
        for private_ref in private_document_refs:
            if private_ref.casefold() in lowered:
                violations.append(f"{relative_path}: private reference {private_ref}")
        if milestone_pattern.search(text):
            violations.append(f"{relative_path}: completed milestone marker")
        for claim in unsupported_claims:
            if claim.casefold() in lowered:
                violations.append(f"{relative_path}: unsupported claim {claim}")

    assert not violations, "\n".join(violations)


def test_tracking_copy_states_the_short_term_identity_limit():
    tracking_source = _read("weedtrack/track.py")

    assert (
        "ByteTrack associates short-term IDs across adjacent frames" in tracking_source
    )
    assert "does not prove long-term identity stability" in tracking_source
