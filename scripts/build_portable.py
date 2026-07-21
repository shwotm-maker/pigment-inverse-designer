"""Build a self-contained portable Windows distribution (no Python install needed).

Produces ``dist/PigmentInverseDesigner/`` (embedded Python 3.11 + all deps + app +
pretrained model) and zips it to ``dist/PigmentInverseDesigner.zip``.

End users just extract the ZIP and double-click ``실행.bat``.

Usage
-----
    python scripts/build_portable.py
    python scripts/build_portable.py --no-zip      # keep folder only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_DIR = DIST / "PigmentInverseDesigner"
PY_DIR = APP_DIR / "python"

PY_VERSION = "3.11.9"
EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GETPIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# App payload copied into the portable folder.
COPY_FILES = ["app.py", "requirements.txt", "README.md"]
COPY_DIRS = ["src", "scripts"]

RUN_BAT = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo   Pigment Inverse Designer  (portable)
echo   브라우저가 자동으로 열립니다. 창을 닫지 마세요.
echo ================================================
python\\python.exe -m streamlit run app.py
pause
"""

TRAIN_BAT = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 데이터 다운로드 시도 후 모델을 재학습합니다...
python\\python.exe -m scripts.download_data
python\\python.exe -m scripts.train_model
pause
"""


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _run(args: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(str(a) for a in args))
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)


def build_embedded_python() -> None:
    """Download the embeddable Python and enable site-packages + pip."""
    PY_DIR.mkdir(parents=True, exist_ok=True)
    embed_zip = DIST / f"python-{PY_VERSION}-embed.zip"
    if not embed_zip.exists():
        _download(EMBED_URL, embed_zip)
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(PY_DIR)

    # Enable `import site` and add Lib\site-packages so pip-installed packages load.
    pth = next(PY_DIR.glob("python*._pth"))
    lines = pth.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for ln in lines:
        out.append("import site" if ln.strip() == "#import site" else ln)
    if "Lib\\site-packages" not in out:
        out.append("Lib\\site-packages")
    # ".." puts the app root (one level above python/) on sys.path so that
    # `import src` and `scripts.*` resolve regardless of the current directory.
    if ".." not in out:
        out.append("..")
    if "import site" not in out:
        out.append("import site")
    pth.write_text("\n".join(out) + "\n", encoding="utf-8")

    # Bootstrap pip.
    getpip = DIST / "get-pip.py"
    if not getpip.exists():
        _download(GETPIP_URL, getpip)
    _run([str(PY_DIR / "python.exe"), str(getpip), "--no-warn-script-location"])


def install_dependencies() -> None:
    _run([
        str(PY_DIR / "python.exe"), "-m", "pip", "install",
        "--no-warn-script-location", "-r", str(ROOT / "requirements.txt"),
    ])


def copy_app_payload() -> None:
    for name in COPY_FILES:
        shutil.copy2(ROOT / name, APP_DIR / name)
    for name in COPY_DIRS:
        dst = APP_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ROOT / name, dst, ignore=shutil.ignore_patterns("__pycache__"))

    # Ship data folders. The sample CSV is library-version independent, so we
    # copy it; the model itself is (re)trained inside the embedded Python below
    # to avoid cross-version pickle issues.
    for sub in ["data/raw", "data/processed", "models", "outputs"]:
        (APP_DIR / sub).mkdir(parents=True, exist_ok=True)
    sample = ROOT / "data" / "raw" / "sample_chromophores.csv"
    if sample.exists():
        shutil.copy2(sample, APP_DIR / "data" / "raw" / sample.name)

    (APP_DIR / "실행.bat").write_text(RUN_BAT, encoding="utf-8")
    (APP_DIR / "데이터갱신_재학습.bat").write_text(TRAIN_BAT, encoding="utf-8")
    (APP_DIR / "사용법.txt").write_text(
        "Pigment Inverse Designer (포터블)\n\n"
        "1) 이 폴더의 '실행.bat' 을 더블클릭하세요.\n"
        "2) 잠시 후 웹브라우저가 자동으로 열립니다 (안 열리면 http://localhost:8501).\n"
        "3) 종료하려면 검은 명령창을 닫으면 됩니다.\n\n"
        "* 파이썬을 따로 설치할 필요가 없습니다.\n"
        "* 최신 공개 데이터로 갱신하려면 '데이터갱신_재학습.bat' 을 실행하세요(인터넷 필요).\n"
        "* (선택) AI 도우미: 앱 사이드바에 ANTHROPIC_API_KEY 를 입력하면 자연어 입력과\n"
        "  결과 해석 리포트를 쓸 수 있습니다. 키가 없어도 예측/검색/생성 기능은 정상 동작합니다.\n"
        "* 본 프로그램은 연구용 개념검증 도구이며 실제 색상/합성/안전성을 보증하지 않습니다.\n",
        encoding="utf-8",
    )


def train_model_in_bundle() -> None:
    """Train the baseline model with the *embedded* Python so the pickled model
    matches the exact numpy/scikit-learn versions shipped in the bundle."""
    _run([str(PY_DIR / "python.exe"), "-m", "scripts.train_model"], cwd=APP_DIR)


def make_zip() -> Path:
    zip_path = DIST / "PigmentInverseDesigner.zip"
    if zip_path.exists():
        zip_path.unlink()
    print("  zipping ...")
    base = shutil.make_archive(str(DIST / "PigmentInverseDesigner"), "zip",
                               root_dir=DIST, base_dir="PigmentInverseDesigner")
    return Path(base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portable Windows distribution.")
    parser.add_argument("--no-zip", action="store_true", help="Skip final ZIP step.")
    args = parser.parse_args()

    if APP_DIR.exists():
        shutil.rmtree(APP_DIR)
    DIST.mkdir(exist_ok=True)

    print("[1/4] Embedded Python ...")
    build_embedded_python()
    print("[2/4] Installing dependencies ...")
    install_dependencies()
    print("[3/5] Copying app payload ...")
    copy_app_payload()
    print("[4/5] Training baseline model inside the bundle ...")
    train_model_in_bundle()
    if args.no_zip:
        print(f"Done (folder): {APP_DIR}")
        return 0
    print("[5/5] Creating ZIP ...")
    zip_path = make_zip()
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {zip_path}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
