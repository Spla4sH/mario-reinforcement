"""Test für die Umgebungsvariablen-Overrides in config."""

import importlib

import config as config_module


def test_env_overrides_apply(monkeypatch):
    monkeypatch.setenv("LEARNING_RATE", "0.001")
    monkeypatch.setenv("EPSILON_DECAY", "500000")
    monkeypatch.setenv("REWARD_CLIP", "1.0")
    try:
        importlib.reload(config_module)
        assert config_module.LEARNING_RATE == 0.001
        assert config_module.EPSILON_DECAY == 500_000
        assert config_module.REWARD_CLIP == 1.0
    finally:
        # Defaults wiederherstellen, damit andere Tests sauberes config sehen
        monkeypatch.delenv("LEARNING_RATE", raising=False)
        monkeypatch.delenv("EPSILON_DECAY", raising=False)
        monkeypatch.delenv("REWARD_CLIP", raising=False)
        importlib.reload(config_module)


def test_defaults_without_env():
    importlib.reload(config_module)
    assert config_module.REWARD_SCALE == 1.0
    assert config_module.REWARD_CLIP is None
    assert config_module.CHECKPOINT_DIR == "checkpoints"
    assert config_module.HIGHLIGHT_DIR == "highlights"


def test_output_dir_overrides(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", "checkpoints_sweep_a")
    monkeypatch.setenv("HIGHLIGHT_DIR", "highlights_sweep_a")
    try:
        importlib.reload(config_module)
        assert config_module.CHECKPOINT_DIR == "checkpoints_sweep_a"
        assert config_module.HIGHLIGHT_DIR == "highlights_sweep_a"
    finally:
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        monkeypatch.delenv("HIGHLIGHT_DIR", raising=False)
        importlib.reload(config_module)


def test_action_set_default_und_down(monkeypatch):
    """ACTION_SET steuert den Aktionsraum: 7 Aktionen (Default) bzw. 8 mit 'down'.

    'down' ist noetig fuer die Roehren in 8-4 – ohne die Aktion waere der
    richtige Weg dort gar nicht erreichbar.
    """
    import importlib

    import config

    monkeypatch.delenv("ACTION_SET", raising=False)
    importlib.reload(config)
    assert config.ACTION_SET == "simple"

    monkeypatch.setenv("ACTION_SET", "down")
    importlib.reload(config)
    assert config.ACTION_SET == "down"

    monkeypatch.delenv("ACTION_SET", raising=False)
    importlib.reload(config)


def test_create_env_signatur_kennt_action_set():
    """create_env muss den Aktionsraum pro Aufruf setzen koennen.

    In der Demo-App laufen Modelle mit 7 und mit 8 Aktionen im selben Prozess –
    eine Umgebungsvariable kann das nicht unterscheiden.

    Geprueft wird per AST statt per Import: ``wrappers`` zieht OpenCV nach, das in
    der CI bewusst fehlt (die Tests sollen ohne Emulator und ohne GPU laufen).
    """
    import ast
    import pathlib

    quelle = pathlib.Path(__file__).resolve().parent.parent / "wrappers.py"
    baum = ast.parse(quelle.read_text(encoding="utf-8"))
    funktionen = {k.name: k for k in ast.walk(baum) if isinstance(k, ast.FunctionDef)}
    assert "create_env" in funktionen

    args = funktionen["create_env"].args
    namen = [a.arg for a in args.args]
    assert "action_set" in namen
    # Default None = "nimm config.ACTION_SET", damit bestehende Aufrufe gleich bleiben
    default = args.defaults[namen.index("action_set") - (len(namen) - len(args.defaults))]
    assert isinstance(default, ast.Constant) and default.value is None


def test_bitmaske_zu_aktion():
    """Menschliche Tastenkombos werden auf die 8 Agenten-Aktionen abgebildet.

    Der Mensch spielt am rohen Env mit allen NES-Kombos, der Agent kennt nur acht.
    Kombos ohne Entsprechung fallen auf ihren Hauptknopf zurueck – bei links+springen
    ist das der Sprung, weil die Traegheit die Richtung ohnehin traegt.
    """
    import importlib.util
    import pathlib

    quelle = pathlib.Path(__file__).resolve().parent.parent / "human_vs_ki.py"
    spec = importlib.util.spec_from_file_location("hvk", quelle)
    hvk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hvk)

    assert hvk._bitmaske_zu_aktion(0x00) == 0      # nichts
    assert hvk._bitmaske_zu_aktion(0x80) == 1      # rechts
    assert hvk._bitmaske_zu_aktion(0x81) == 2      # rechts + A
    assert hvk._bitmaske_zu_aktion(0x83) == 4      # rechts + A + B
    assert hvk._bitmaske_zu_aktion(0x20) == 7      # runter (Roehren!)
    assert hvk._bitmaske_zu_aktion(0x41) == 5      # links + A -> Sprung
