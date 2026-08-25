"""Gate for the command-line entry point, and for the one-word launcher.

Nothing here starts a real server: the serve path is intercepted, and the
only sockets bound are ones this file owns and closes. The point of the
suite is the *dispatch* -- that an empty argv means "run the web app", that
every explicit subcommand still means what it used to, and that the three
helpers the launcher leans on (free-port scan, readiness poll, desktop
shortcut) behave when the world is uncooperative.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

from heliostat import __version__, cli


@pytest.fixture
def recorded_serve(monkeypatch):
    """Replace the serve path with a recorder; returns the call list."""
    calls: list[dict] = []

    def _fake(host, port, open_browser):
        calls.append({"host": host, "port": port, "open_browser": open_browser})
        return 0

    monkeypatch.setattr(cli, "_serve", _fake)
    return calls


def _bound_socket(port: int = 0) -> socket.socket:
    """A listening socket on 127.0.0.1, owning ``port`` for the test's lifetime."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_no_args_runs_the_web_app(recorded_serve):
    """A bare ``heliostat`` (and a double-clicked .exe) launches the app."""
    assert cli.main([]) == 0
    assert recorded_serve == [
        # port None is "the default port, or the next free one" -- resolved
        # inside _serve so the no-args path cannot fail on a busy 8420.
        {"host": cli.DEFAULT_HOST, "port": None, "open_browser": True}
    ]
    assert cli.DEFAULT_PORT == 8420


def test_no_args_serves_the_default_port_and_arms_the_browser(monkeypatch):
    """End to end through the real ``_serve``, with only uvicorn stubbed out."""
    pytest.importorskip("fastapi")
    if not cli.port_is_free(cli.DEFAULT_HOST, cli.DEFAULT_PORT):
        pytest.skip(f"port {cli.DEFAULT_PORT} is busy on this machine")

    served: dict = {}
    browsed: list[tuple] = []

    def _fake_run(app, host, port):
        served["host"], served["port"] = host, port

    monkeypatch.setattr("heliostat.web.app.create_app", lambda: "app")
    monkeypatch.setattr(cli, "_open_browser_when_ready", lambda *a: browsed.append(a))
    monkeypatch.setitem(sys.modules, "uvicorn", type("U", (), {"run": staticmethod(_fake_run)}))

    assert cli.main([]) == 0
    assert served == {"host": cli.DEFAULT_HOST, "port": cli.DEFAULT_PORT}

    # The opener runs on its own thread, so give it a moment to land.
    deadline = time.monotonic() + 5.0
    while not browsed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert browsed == [(f"http://{cli.DEFAULT_HOST}:{cli.DEFAULT_PORT}/", cli.DEFAULT_HOST, 8420)]


def test_help_still_prints_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    assert "usage: heliostat" in capsys.readouterr().out


def test_short_help_flag_is_not_the_launcher(recorded_serve, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["-h"])
    assert excinfo.value.code == 0
    assert "usage: heliostat" in capsys.readouterr().out
    assert recorded_serve == []


def test_version_still_prints_the_version(recorded_serve, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
    assert recorded_serve == []


def test_serve_subcommand_matches_the_no_args_path(recorded_serve):
    assert cli.main(["serve"]) == 0
    assert recorded_serve == [{"host": cli.DEFAULT_HOST, "port": None, "open_browser": True}]


@pytest.mark.parametrize("flag", ["--no-browser", "--no-open"])
def test_serve_can_opt_out_of_the_browser(recorded_serve, flag):
    assert cli.main(["serve", flag]) == 0
    assert recorded_serve[0]["open_browser"] is False


def test_serve_honours_explicit_host_and_port(recorded_serve):
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9123"]) == 0
    assert recorded_serve == [{"host": "0.0.0.0", "port": 9123, "open_browser": True}]


def test_layout_subcommand_is_untouched(recorded_serve, tmp_path):
    out = tmp_path / "field.csv"
    assert cli.main(["layout", "fermat", "--n", "12", "--a", "4.5", "-o", str(out)]) == 0
    assert out.exists()
    assert recorded_serve == []


def test_subcommand_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["trace", "--help"])
    assert excinfo.value.code == 0
    assert "--field" in capsys.readouterr().out


# --------------------------------------------------------------------------
# port selection
# --------------------------------------------------------------------------


def test_free_port_scan_steps_over_a_busy_port():
    with _bound_socket() as sock:
        busy = sock.getsockname()[1]
        assert cli.port_is_free("127.0.0.1", busy) is False
        assert cli.find_free_port("127.0.0.1", start=busy, last=busy + 5) == busy + 1


def test_free_port_scan_gives_up_with_a_clear_error():
    with _bound_socket() as sock:
        busy = sock.getsockname()[1]
        with pytest.raises(cli.NoFreePortError, match=f"between {busy} and {busy}"):
            cli.find_free_port("127.0.0.1", start=busy, last=busy)


def test_explicit_busy_port_is_an_error_not_a_silent_move():
    with _bound_socket() as sock:
        busy = sock.getsockname()[1]
        with pytest.raises(cli.PortBusyError, match=f"port {busy} is already in use"):
            cli.resolve_port("127.0.0.1", busy)


def test_explicit_free_port_is_used_as_asked():
    with _bound_socket() as sock:
        free = sock.getsockname()[1]
    assert cli.resolve_port("127.0.0.1", free) == free


def test_serve_reports_a_busy_explicit_port(capsys, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setitem(
        sys.modules, "uvicorn", type("U", (), {"run": staticmethod(lambda *a, **k: None)})
    )
    with _bound_socket() as sock:
        busy = sock.getsockname()[1]
        assert cli._serve("127.0.0.1", busy, open_browser=False) == 1
    err = capsys.readouterr().err
    assert f"port {busy} is already in use" in err
    assert "--port" in err


# --------------------------------------------------------------------------
# browser auto-open
# --------------------------------------------------------------------------


def test_browser_opens_once_and_only_after_the_port_accepts():
    opened: list[str] = []
    polls: list[float] = []
    listener: list[socket.socket] = []

    with _bound_socket() as probe:
        port = probe.getsockname()[1]
    # `port` is now free: the poll must not open a browser onto it yet.

    def sleeper(delay: float) -> None:
        polls.append(delay)
        if len(polls) == 3:
            listener.append(_bound_socket(port))
        # The browser must not have been opened during any failed poll.
        assert opened == []

    try:
        assert (
            cli._open_browser_when_ready(
                "http://127.0.0.1/",
                "127.0.0.1",
                port,
                timeout=10.0,
                interval=0.0,
                opener=opened.append,
                sleeper=sleeper,
            )
            is True
        )
    finally:
        for sock in listener:
            sock.close()

    assert opened == ["http://127.0.0.1/"]
    assert len(polls) == 3


def test_browser_gives_up_cleanly_after_the_timeout():
    opened: list[str] = []
    with _bound_socket() as probe:
        port = probe.getsockname()[1]

    assert (
        cli._open_browser_when_ready(
            "http://127.0.0.1/",
            "127.0.0.1",
            port,
            timeout=0.05,
            interval=0.01,
            opener=opened.append,
        )
        is False
    )
    assert opened == []


def test_wildcard_bind_is_polled_and_opened_on_loopback():
    assert cli._connect_host("0.0.0.0") == "127.0.0.1"
    assert cli._connect_host("::") == "127.0.0.1"
    assert cli._connect_host("192.168.1.5") == "192.168.1.5"


# --------------------------------------------------------------------------
# banner
# --------------------------------------------------------------------------


def test_banner_names_the_url_the_browser_and_the_off_switch():
    text = cli._banner("http://127.0.0.1:8420/", opening_browser=True, moved_from=None)
    lines = text.splitlines()
    assert lines[0] == f"heliostat {__version__} — starting the web app"
    assert lines[1] == "  http://127.0.0.1:8420/  (opening your browser)"
    assert "Ctrl+C" in text
    assert "heliostat --help" in text
    assert "busy" not in text


def test_banner_says_when_the_default_port_was_busy():
    text = cli._banner("http://127.0.0.1:8421/", opening_browser=False, moved_from=8420)
    assert "port 8420 was busy" in text
    assert "opening your browser" not in text


@pytest.mark.parametrize("encoding", ["ascii", "cp1252", None, ""])
def test_banner_degrades_unless_the_stream_is_utf8(monkeypatch, encoding):
    """cp1252 can encode an em dash but is usually read back as mojibake, and
    a redirected or frozen stream may report no encoding at all."""

    class _Stream:
        pass

    stream = _Stream()
    stream.encoding = encoding
    monkeypatch.setattr(sys, "stdout", stream)
    assert cli._console_safe("a — b") == "a - b"


@pytest.mark.parametrize("encoding", ["utf-8", "UTF8"])
def test_banner_keeps_its_dash_on_a_utf8_stream(monkeypatch, encoding):
    class _Stream:
        pass

    stream = _Stream()
    stream.encoding = encoding
    monkeypatch.setattr(sys, "stdout", stream)
    assert cli._console_safe("a — b") == "a — b"


# --------------------------------------------------------------------------
# missing web extra
# --------------------------------------------------------------------------


def test_missing_web_extra_is_two_friendly_lines_not_a_traceback(monkeypatch, capsys):
    # `import uvicorn` against a None entry raises ImportError, which is what
    # an install without the [web] extra does for real.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    code = cli.main([])
    assert code == 1
    captured = capsys.readouterr()
    lines = captured.err.strip().splitlines()
    assert len(lines) == 2
    assert "web" in lines[0] and "not installed" in lines[0]
    assert 'pip install "heliostat[web]"' in lines[1]
    assert captured.out == ""


# --------------------------------------------------------------------------
# desktop shortcut
# --------------------------------------------------------------------------


def test_macos_launcher_is_a_two_line_exec_script():
    text = cli.macos_command_text(Path("/opt/venv/bin/heliostat"))
    assert text.splitlines() == ["#!/bin/sh", 'exec "/opt/venv/bin/heliostat"']


def test_linux_launcher_keeps_a_terminal_for_the_off_switch():
    text = cli.linux_desktop_text(Path("/opt/venv/bin/heliostat"))
    assert text.startswith("[Desktop Entry]\n")
    assert 'Exec="/opt/venv/bin/heliostat"' in text
    # Terminal=true is the whole reason a user can stop the server again.
    assert "Terminal=true" in text
    assert "Type=Application" in text


def test_posix_launcher_refuses_to_clobber(tmp_path):
    target = tmp_path / "Heliostat.command"
    cli._write_posix_launcher(target, "first\n", force=False)
    with pytest.raises(cli.ShortcutError, match="--force"):
        cli._write_posix_launcher(target, "second\n", force=False)
    cli._write_posix_launcher(target, "second\n", force=True)
    assert target.read_text() == "second\n"


def test_posix_launcher_rejects_a_missing_directory(tmp_path):
    with pytest.raises(cli.ShortcutError, match="not a directory"):
        cli._write_posix_launcher(tmp_path / "nope" / "x.command", "x\n", force=False)


def test_executable_falls_back_to_the_running_interpreter(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    bindir = tmp_path / "Scripts"
    bindir.mkdir()
    name = "heliostat.exe" if sys.platform == "win32" else "heliostat"
    (bindir / name).write_text("")
    monkeypatch.setattr(sys, "executable", str(bindir / "python.exe"))
    assert cli.heliostat_executable() == bindir / name


def test_shortcut_reports_a_missing_executable(monkeypatch, capsys):
    monkeypatch.setattr(cli, "heliostat_executable", lambda: None)
    assert cli.main(["shortcut"]) == 1
    err = capsys.readouterr().err
    assert 'pip install "heliostat[web]"' in err


def _lnk_target(path: Path) -> str:
    """Read a .lnk back through the same COM object that wrote it."""
    import subprocess

    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
        f"'{path}'); Write-Output $s.TargetPath"
    )
    proc = subprocess.run(
        [cli._powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.skipif(sys.platform != "win32", reason="the .lnk writer is Windows-only")
def test_windows_shortcut_is_created_and_points_at_the_executable(tmp_path, capsys):
    assert cli.main(["shortcut", "--path", str(tmp_path)]) == 0
    lnk = tmp_path / "heliostat.lnk"
    assert lnk.is_file()
    exe = cli.heliostat_executable()
    assert exe is not None
    assert _lnk_target(lnk).lower() == str(exe).lower()

    out = capsys.readouterr().out
    assert str(lnk) in out
    assert str(exe) in out


@pytest.mark.skipif(sys.platform != "win32", reason="the .lnk writer is Windows-only")
def test_windows_shortcut_refuses_to_clobber_without_force(tmp_path, capsys):
    assert cli.main(["shortcut", "--path", str(tmp_path)]) == 0
    capsys.readouterr()

    assert cli.main(["shortcut", "--path", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "already exists" in err and "--force" in err

    assert cli.main(["shortcut", "--path", str(tmp_path), "--force"]) == 0
    assert (tmp_path / "heliostat.lnk").is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="the .lnk writer is Windows-only")
def test_windows_shortcut_rejects_a_missing_directory(tmp_path, capsys):
    assert cli.main(["shortcut", "--path", str(tmp_path / "nope")]) == 1
    assert "not a directory" in capsys.readouterr().err
