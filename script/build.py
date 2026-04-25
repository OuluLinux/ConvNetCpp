#!/usr/bin/env python3

import os
import re
import shutil
import subprocess
import sys
import tempfile
import platform
import datetime
from pathlib import Path


def print_help(prog):
    print(
        "\n".join(
            [
                f"Usage: {prog} [options] <target>",
                f"       {prog} tutorial",
                "",
                "Options:",
                "  --conf-release, -ConfRelease, -cr  Select Release mainconfig",
                "  --conf-debug, -ConfDebug, -cd      Select Debug mainconfig",
                "  --mainconf X, -mc X                Select mainconfig by index or name",
                "  --list-conf P, -lc P               List mainconfigs for a package",
                "  --release, -Release, -r            Use release build flags in umk",
                "  --method X, -m X                   Select build method by index, name, or path",
                "  --list-methods, -ListMethods       List build methods and exit",
                "  --android, -Android                Select Android build method (BUILDER=ANDROID)",
                "  --bootstrap, -Bootstrap, -bs       Bootstrap-build umk via Makefile",
                "  --smoketest, -Smoketest            Build a single source file (UWP smoke test)",
                "  --clean, -Clean, -c                Clean build",
                "  --jobs N, -j N, -jN                Parallel jobs",
                "  --verbose, -Verbose, -v            Verbose output",
                "  --dump-cmd                         Dump the umk command and exit",
                "  --help, -Help                      Show this help",
            ]
        )
    )


def print_tutorial(prog):
    print(
        "\n".join(
            [
                "Build tutorial",
                "",
                "Common usage:",
                f"  {prog} --conf-release --release Classify2D",
                f"  {prog} -cr -r Classify2D",
                f"  {prog} --conf-debug --release OCRDatasetEditor",
                f"  {prog} -cd -r OCRTraining",
                "",
                "Select a mainconfig explicitly:",
                f"  {prog} --list-conf Classify2D",
                f"  {prog} --mainconf 0 Classify2D",
                f"  {prog} --mainconf \"Release (Posix)\" OCRTraining",
                "",
                "Control method selection:",
                f"  {prog} --list-methods",
                f"  {prog} --method 0 Classify2D",
                "",
                "Bootstrap umk:",
                f"  {prog} --bootstrap",
            ]
        )
    )


def parse_args(argv):
    opts = {
        "conf_mode": None,
        "release": False,
        "clean": False,
        "jobs": None,
        "verbose": False,
        "method": None,
        "list_methods": False,
        "list_conf": None,
        "mainconf": None,
        "android": False,
        "bootstrap": False,
        "smoketest": False,
        "target": None,
        "dump_cmd": False,
        "help": False,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        lower = arg.lower()
        if lower in ("--help", "-help", "-h"):
            opts["help"] = True
            i += 1
            continue
        if lower in ("--conf-release", "-confrelease", "-cr"):
            opts["conf_mode"] = "release"
            i += 1
            continue
        if lower in ("--conf-debug", "-confdebug", "-cd"):
            opts["conf_mode"] = "debug"
            i += 1
            continue
        if lower in ("--list-conf", "-list-conf", "-lc"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for list-conf")
            i += 1
            opts["list_conf"] = argv[i]
            i += 1
            continue
        if lower in ("--mainconf", "-mainconf", "-mc"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for mainconf")
            i += 1
            opts["mainconf"] = argv[i]
            i += 1
            continue
        if lower.startswith("--mainconf="):
            opts["mainconf"] = arg.split("=", 1)[1]
            i += 1
            continue
        if lower.startswith("-mc") and len(arg) > 3:
            opts["mainconf"] = arg[3:]
            i += 1
            continue
        if lower in ("--release", "-release", "-r"):
            opts["release"] = True
            i += 1
            continue
        if lower in ("--list-methods", "-listmethods"):
            opts["list_methods"] = True
            i += 1
            continue
        if lower in ("--android", "-android"):
            opts["android"] = True
            i += 1
            continue
        if lower in ("--bootstrap", "-bootstrap", "-bs"):
            opts["bootstrap"] = True
            i += 1
            continue
        if lower in ("--smoketest", "-smoketest"):
            opts["smoketest"] = True
            i += 1
            continue
        if lower in ("--method", "-method", "-m"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for method")
            i += 1
            opts["method"] = argv[i]
            i += 1
            continue
        if lower.startswith("--method="):
            opts["method"] = arg.split("=", 1)[1]
            i += 1
            continue
        if lower.startswith("-m") and len(arg) > 2:
            opts["method"] = arg[2:]
            i += 1
            continue
        if lower in ("--clean", "-clean", "-c"):
            opts["clean"] = True
            i += 1
            continue
        if lower in ("--verbose", "-verbose", "-v"):
            opts["verbose"] = True
            i += 1
            continue
        if lower in ("--dump-cmd", "-dump-cmd"):
            opts["dump_cmd"] = True
            i += 1
            continue
        if lower in ("--jobs", "-jobs", "-j"):
            if i + 1 >= len(argv):
                raise ValueError("Missing value for jobs")
            i += 1
            opts["jobs"] = parse_jobs(argv[i])
            i += 1
            continue
        if lower.startswith("--jobs="):
            opts["jobs"] = parse_jobs(arg.split("=", 1)[1])
            i += 1
            continue
        if lower.startswith("-j") and len(arg) > 2:
            opts["jobs"] = parse_jobs(arg[2:])
            i += 1
            continue
        if arg.startswith("-"):
            raise ValueError(f"Unknown option: {arg}")
        if opts["target"] is not None:
            raise ValueError("Only one target is supported")
        opts["target"] = arg
        i += 1
    return opts


def parse_jobs(value):
    try:
        jobs = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid jobs value: {value}") from exc
    if jobs <= 0:
        raise ValueError("Jobs must be positive")
    return jobs


def find_repo_root():
    return Path(__file__).resolve().parent.parent


def resolve_upp_path(repo_root, target):
    target_path = Path(target)
    if target_path.is_dir():
        candidates = list(target_path.glob("*.upp"))
        if len(candidates) == 1:
            return candidates[0].resolve()
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple .upp files in {target_path}: "
                + ", ".join(str(path) for path in candidates)
            )
        raise ValueError(f"No .upp file found in {target_path}")
    if target_path.suffix == ".upp":
        path = target_path
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise ValueError(f"Missing .upp file: {path}")
        return path.resolve()
    return find_upp_by_name(repo_root, f"{target}.upp")


def find_upp_by_name(repo_root, filename):
    search_roots = [
        repo_root / "examples",
        repo_root / "tests",
        repo_root / "src",
        (repo_root.parent / "ai-upp" / "uppsrc").resolve(),  # External uppsrc
    ]
    matches = []
    for root in search_roots:
        if not root.exists():
            continue
        matches.extend(root.rglob(filename))
    if not matches:
        for root, dirs, files in os.walk(repo_root):
            if ".git" in dirs:
                dirs.remove(".git")
            if "bin" in dirs:
                dirs.remove("bin")
            if filename in files:
                matches.append(Path(root) / filename)
    if not matches:
        raise ValueError(f"Unable to locate .upp file for {filename}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple .upp files named {filename}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def read_mainconfigs(upp_path):
    text = upp_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "mainconfig":
            start = idx + 1
            break
    if start is None:
        return []
    block_lines = []
    for line in lines[start:]:
        if not line.strip():
            if block_lines:
                break
            continue
        if line[:1].isspace():
            block_lines.append(line)
            continue
        break
    block = "\n".join(block_lines)
    entries = re.findall(r'"([^"]*)"\s*=\s*"([^"]*)"', block)
    return [{"name": name, "flags": flags} for name, flags in entries]


def config_matches_os(name, is_windows):
    lower = name.lower()
    has_windows = "windows" in lower
    has_posix = "posix" in lower
    if has_windows and not is_windows:
        return False
    if has_posix and is_windows:
        return False
    return True


def select_config(configs, conf_mode, is_windows):
    eligible = [cfg for cfg in configs if config_matches_os(cfg["name"], is_windows)]
    if not configs:
        return None, None
    if conf_mode is None:
        if eligible:
            return eligible[0], configs.index(eligible[0])
        return None, None
    mode = conf_mode.lower()
    for cfg in eligible:
        name_lower = cfg["name"].lower()
        if mode == "release" and "release" in name_lower:
            return cfg, configs.index(cfg)
        if mode == "debug" and "debug" in name_lower:
            return cfg, configs.index(cfg)
    return None, None


def normalize_flags(flags):
    return ",".join(split_flags(flags))


def split_flags(flags):
    tokens = [token for token in re.split(r"[\s,]+", flags.strip()) if token]
    normalized = []
    for token in tokens:
        token = token.strip(",;")
        if not token:
            continue
        normalized.append(token)
    return normalized


def apply_debug_full(flags, release):
    if not flags:
        return []
    normalized = split_flags(flags) if isinstance(flags, str) else list(flags)
    filtered = []
    for flag in normalized:
        check = flag.lstrip(".")
        if check in ("DEBUG_FULL", "FULL_DEBUG"):
            continue
        filtered.append(flag)
    if not release:
        filtered.append("DEBUG_FULL")
    return filtered


def select_config_by_token(configs, token):
    token = token.strip()
    try:
        index = int(token, 10)
    except ValueError:
        index = None
    if index is not None:
        if index < 0 or index >= len(configs):
            raise ValueError(f"Mainconfig index out of range: {index}")
        return configs[index], index
    for idx, cfg in enumerate(configs):
        if cfg["name"].lower() == token.lower():
            return cfg, idx
    raise ValueError(f"Unknown mainconfig: {token}")


def method_dirs_posix():
    base = Path("~/.config/u++").expanduser()
    return [base / "theide", base / "umk"]


def collect_methods_posix():
    auto_methods = collect_auto_methods()
    methods = []
    seen = set()
    methods.extend(auto_methods)
    seen.update(method["path"].resolve() for method in auto_methods)
    for method_dir in method_dirs_posix():
        if not method_dir.exists():
            continue
        for path in method_dir.glob("*.bm"):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            methods.append(
                {
                    "name": path.stem,
                    "display": path.stem,
                    "path": path,
                    "generated": False,
                    "auto": False,
                    "builder": read_bm_builder(path),
                }
            )
    return methods


def auto_method_dir():
    return Path(tempfile.gettempdir()) / "upp_build_methods"


def collect_auto_methods():
    ensure_auto_methods()
    methods = []
    method_dir = auto_method_dir()
    if not method_dir.exists():
        return methods
    for path in sorted(method_dir.glob("*.bm")):
        name = path.stem
        display = f"{name} (auto)"
        methods.append(
            {
                "name": f"auto-{name.lower()}",
                "display": display,
                "path": path,
                "generated": True,
                "auto": True,
                "builder": None,
            }
        )
    return methods


def ensure_auto_methods():
    method_dir = auto_method_dir()
    method_dir.mkdir(parents=True, exist_ok=True)

    def ensure_method(name, compiler):
        if not shutil.which(compiler):
            return
        path = method_dir / f"{name}.bm"
        content = build_method_template(name, compiler)
        path.write_text(content, encoding="utf-8")

    ensure_method("CLANG", "clang++")
    ensure_method("GCC", "g++")


def build_method_template(name, compiler):
    paths = detect_build_paths()
    path_line = ";".join(paths["path"])
    include_line = ";".join(paths["include"])
    lib_line = ";".join(paths["lib"])
    return "\n".join(
        [
            f'BUILDER = "{name}";',
            f'COMPILER = "{compiler}";',
            'COMMON_OPTIONS = "-mpopcnt";',
            'COMMON_CPP_OPTIONS = "-std=c++17 -Wno-logical-op-parentheses";',
            'COMMON_C_OPTIONS = "";',
            'COMMON_LINK = "";',
            'COMMON_FLAGS = "";',
            'DEBUG_INFO = "2";',
            'DEBUG_BLITZ = "1";',
            'DEBUG_LINKMODE = "1";',
            'DEBUG_OPTIONS = "-O0";',
            'DEBUG_FLAGS = "";',
            'DEBUG_LINK = "";',
            'RELEASE_BLITZ = "1";',
            'RELEASE_LINKMODE = "1";',
            'RELEASE_OPTIONS = "-O3 -ffunction-sections -fdata-sections";',
            'RELEASE_FLAGS = "";',
            'RELEASE_LINK = "-Wl,--gc-sections";',
            'DEBUGGER = "gdb";',
            'ALLOW_PRECOMPILED_HEADERS = "0";',
            'DISABLE_BLITZ = "0";',
            f'PATH = "{path_line}";',
            f'INCLUDE = "{include_line}";',
            f'LIB = "{lib_line}";',
            'LINKMODE_LOCK = "0";',
            "",
        ]
    )


def detect_build_paths():
    path_entries = []
    include_entries = []
    lib_entries = []

    def add_path(entry, collection):
        if entry and entry.exists():
            value = str(entry)
            if value not in collection:
                collection.append(value)

    add_path(Path("/usr/bin"), path_entries)

    include_candidates = [
        Path("/usr/include"),
        Path("/usr/local/include"),
    ]
    for entry in include_candidates:
        add_path(entry, include_entries)

    lib_candidates = [
        Path("/usr/lib64"),
        Path("/usr/lib"),
        Path("/usr/local/lib"),
        Path("/usr/local/lib64"),
    ]
    for entry in lib_candidates:
        add_path(entry, lib_entries)

    return {"path": path_entries, "include": include_entries, "lib": lib_entries}


def select_posix_default_method(methods):
    if not methods:
        return None
    non_auto = [method for method in methods if not method.get("auto")]
    by_name = {method["name"].lower(): method for method in non_auto}
    for name in ("clang", "gcc"):
        if name in by_name:
            return by_name[name]
    by_auto_name = {method["name"].lower(): method for method in methods}
    for name in ("auto-clang", "auto-gcc"):
        if name in by_auto_name:
            return by_auto_name[name]
    return methods[0]


def read_bm_builder(path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'^\s*BUILDER\s*=\s*"([^"]+)"\s*;', text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def resolve_method(methods, method_arg):
    if method_arg is None:
        return None
    try:
        index = int(method_arg, 10)
    except ValueError:
        index = None
    if index is not None:
        if index < 0 or index >= len(methods):
            raise ValueError(f"Build method index out of range: {index}")
        return methods[index]
    method_path = Path(method_arg).expanduser()
    if method_path.suffix == ".bm" or method_path.is_absolute() or os.sep in method_arg:
        if method_path.exists():
            return {
                "name": method_path.stem,
                "display": method_path.stem,
                "path": method_path,
                "generated": False,
                "auto": False,
                "builder": read_bm_builder(method_path),
            }
        method_arg = method_path.stem
    for method in methods:
        if method["name"].lower() == method_arg.lower():
            return method
        if method.get("display") and method["display"].lower() == method_arg.lower():
            return method
    lowered = method_arg.lower()
    if lowered in ("clang", "gcc"):
        for method in methods:
            if method["name"].lower() == lowered:
                return method
        for method in methods:
            if method["name"].lower() == f"auto-{lowered}":
                return method
    raise ValueError(f"Unknown build method: {method_arg}")


def list_methods(methods):
    if not methods:
        print("No build methods found.")
        return
    for idx, method in enumerate(methods):
        suffix = " (generated)" if method.get("generated") else ""
        display = method.get("display") or method["name"]
        builder = method.get("builder")
        builder_note = f" [builder: {builder}]" if builder else ""
        print(f"[{idx}] {display}: {method['path']}{suffix}{builder_note}")


def build_command(
    umk_path, roots, target, build_model, build_flags, flags, output_path, jobs, verbose
):
    args = [str(umk_path), roots, target, str(build_model), build_flags]
    if jobs:
        args.append(f"-H{jobs}")
    if flags:
        args.append(f"+{flags}")
    args.append(str(output_path))
    if verbose:
        print("Command:", " ".join(args))
    return args


def resolve_umk_path():
    # Check for umk in PATH first
    umk_in_path = shutil.which("umk")
    if umk_in_path:
        return umk_in_path

    # Check local bin directory
    bin_dir = Path("bin")
    candidates = [bin_dir / "umk", bin_dir / "umk.exe"]
    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "umk executable not found. Please ensure 'umk' is in your PATH or "
        "'bin/umk' exists in the project directory."
    )


def default_build_model(is_windows, methods):
    if is_windows:
        return None  # Not implemented for Windows yet
    method = select_posix_default_method(methods)
    return method


def main():
    try:
        opts = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not sys.argv[1:] or opts["help"]:
        print_help(Path(sys.argv[0]).name)
        return 0
    if opts["list_methods"]:
        is_windows = os.name == "nt"
        methods = collect_methods_posix() if not is_windows else []
        list_methods(methods)
        return 0
    if opts["list_conf"]:
        repo_root = find_repo_root()
        os.chdir(repo_root)
        try:
            upp_path = resolve_upp_path(repo_root, opts["list_conf"])
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        configs = read_mainconfigs(upp_path)
        if not configs:
            print("No mainconfig entries found.")
            return 0
        for idx, cfg in enumerate(configs):
            print(f"[{idx}] {cfg['name']} = {cfg['flags']}")
        return 0

    if not opts["target"]:
        print("Missing target.", file=sys.stderr)
        print_help(Path(sys.argv[0]).name)
        return 2
    if opts["target"].lower() == "tutorial":
        print_tutorial(Path(sys.argv[0]).name)
        return 0

    repo_root = find_repo_root()
    os.chdir(repo_root)

    try:
        upp_path = resolve_upp_path(repo_root, opts["target"])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    target = upp_path.stem
    is_windows = os.name == "nt"
    methods = collect_methods_posix() if not is_windows else []
    configs = read_mainconfigs(upp_path)

    if opts["mainconf"]:
        try:
            selected, index = select_config_by_token(configs, opts["mainconf"])
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        if not opts["release"] and opts["conf_mode"] is None:
            selected, index = select_config(configs, "debug", is_windows)
            if selected:
                opts["conf_mode"] = "debug"
            else:
                selected, index = select_config(configs, None, is_windows)
        else:
            selected, index = select_config(configs, opts["conf_mode"], is_windows)

    flags = ""
    if selected:
        flags = normalize_flags(selected["flags"])

    if flags:
        flags = ",".join(apply_debug_full(flags, opts["release"]))
    elif not opts["release"]:
        flags = "DEBUG_FULL"

    try:
        resolved_method = resolve_method(methods, opts["method"])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    build_method = resolved_method or default_build_model(is_windows, methods)
    if build_method is None:
        print("No suitable build method found.", file=sys.stderr)
        return 2
    build_model = build_method["path"]
    if not build_model.exists():
        print(f"Build model not found: {build_model}", file=sys.stderr)
        return 2

    build_flags = "-rbsH1" if opts["release"] else "-bsdH1"
    if opts["clean"]:
        build_flags += "a"

    # Use external uppsrc path
    ai_upp_root = (repo_root / ".." / "ai-upp").resolve()
    ai_upp_uppsrc = ai_upp_root / "uppsrc"
    ai_upp_bazaar = ai_upp_root / "bazaar"
    
    roots = (
        f".,./examples,./tests,./src,./upptst,"
        f"{str(ai_upp_uppsrc)},{str(Path.home())}/topside-code/bazaar,"
        f"{ai_upp_bazaar}"
    )

    output_name = f"{target}.exe" if is_windows else target
    output_path = Path("bin") / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if opts["verbose"]:
        if selected:
            print(f"Mainconfig: [{index}] {selected['name']}")
        else:
            print("Mainconfig: (none)")
        print(f"Flags: {flags if flags else '(none)'}")
        builder = build_method.get("builder")
        builder_note = f" [{builder}]" if builder else ""
        print(f"Build model: {build_model}{builder_note}")
        print(f"Build flags: {build_flags}")

    try:
        umk_path = resolve_umk_path()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    args = build_command(
        umk_path,
        roots,
        target,
        build_model,
        build_flags,
        flags,
        output_path,
        opts["jobs"],
        opts["verbose"],
    )
    if opts["dump_cmd"]:
        print(" ".join(args))
        return 0
    result = subprocess.run(args)

    if result.returncode != 0:
        return result.returncode

    if output_path.exists():
        print(f"Executable compiled: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
