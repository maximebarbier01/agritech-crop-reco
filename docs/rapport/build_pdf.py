"""Génère reports/rapport_metier.pdf à partir de docs/rapport/rapport_metier.html.

Impression par Chrome en mode sans affichage (« headless »). Sous WSL, le Chrome de
Windows est utilisé ; sinon, un Chrome/Chromium du système. Les captures d'écran
absentes (MLflow, application, CI/CD) sont remplacées par un cadre « capture à
insérer » et listées, pour que le PDF reste générable à tout moment.

Usage : python docs/rapport/build_pdf.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "rapport_metier.html"
OUTPUT = ROOT / "reports" / "rapport_metier.pdf"

CHROME_CANDIDATES = [
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
]
IMG = re.compile(r'<img src="([^"]+)" alt="([^"]*)">')


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    sys.exit("Aucun Chrome / Chromium / Edge trouvé pour imprimer le PDF.")


def is_windows_exe(path: str) -> bool:
    return path.startswith("/mnt/") and path.endswith(".exe")


def to_windows(path: Path) -> str:
    return subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True
    ).stdout.strip()


def with_placeholders(html: str) -> tuple[str, list[str]]:
    """Remplace chaque image manquante par un cadre explicite."""
    missing = []

    def repl(match: re.Match) -> str:
        src, alt = match.group(1), match.group(2)
        if (HERE / src).resolve().exists():
            return match.group(0)
        missing.append(src.removeprefix("../../"))
        return f'<div class="placeholder">Capture à insérer : <b>{alt}</b><br>{missing[-1]}</div>'

    return IMG.sub(repl, html), missing


def main() -> None:
    html, missing = with_placeholders(SOURCE.read_text(encoding="utf-8"))
    chrome = find_chrome()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Fichier HTML de travail à côté de la source (chemins relatifs des images inchangés)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=HERE, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        page = Path(tmp.name)
    profile = Path(tempfile.mkdtemp(prefix="chrome-report-"))
    try:
        if is_windows_exe(chrome):
            # Chrome Windows : chemins Windows, PDF écrit dans un dossier temporaire Windows
            win_tmp = Path("/mnt/c/Users") / os.getenv("WIN_USER", "maxim") / "AppData/Local/Temp"
            pdf_tmp = win_tmp / "agritech_rapport_metier.pdf"
            url = "file:///" + to_windows(page).replace("\\", "/")
            args = [
                f"--print-to-pdf={to_windows(pdf_tmp)}",
                f"--user-data-dir={to_windows(win_tmp / 'chrome-report-profile')}",
            ]
        else:
            pdf_tmp = OUTPUT
            url = page.as_uri()
            args = [f"--print-to-pdf={OUTPUT}", f"--user-data-dir={profile}"]
        pdf_tmp.unlink(missing_ok=True)  # ne pas confondre avec un PDF d'une exécution précédente
        cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", *args, url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not pdf_tmp.exists():
            sys.exit(f"Échec de l'impression PDF :\n{result.stderr[-2000:]}")
        if pdf_tmp != OUTPUT:
            shutil.move(pdf_tmp, OUTPUT)
    finally:
        page.unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)

    print(f"PDF écrit : {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e6:.1f} Mo)")
    if missing:
        print(f"{len(missing)} capture(s) à fournir :")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
