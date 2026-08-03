"""Startet die lokale Gradio-Demo (32 Level im Browser, mit Grad-CAM-Toggle).

    python scripts/app.py

Die eigentliche App liegt in ``mario/demo.py`` – dort, weil ausser diesem Starter
auch ``scripts/gen_demos.py`` und ``scripts/human_vs_ki.py`` die Level-Tabelle und
die Predictoren importieren.

Der oeffentliche Hugging-Face-Space benutzt diese Datei nicht, sondern die
eigenstaendige Kopie in ``deploy/app.py`` (zeigt vorberechnete Videos, braucht
weder Torch noch den Emulator) – siehe DEPLOY.md.
"""

from mario.demo import build

if __name__ == "__main__":
    build().launch()
