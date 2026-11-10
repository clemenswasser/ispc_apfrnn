#!/usr/bin/env python3
"""Project rule checks, run from git hooks and scripts/ci.sh.

  agent_check.py pre-commit        check the staged change
  agent_check.py commit-msg FILE   check a commit message
  agent_check.py range BASE        check BASE..HEAD for review
  agent_check.py log               check .perf-log/experiments.jsonl

Rules that only make sense for unattended work apply when an agent is an
ancestor of this process. People can bypass a hook with --no-verify.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROTECTED = (
    "AGENTS.md",
    "CLAUDE.md",
    "opencode.json",
    ".claude/",
    ".githooks/",
    "scripts/agent_check.py",
    "scripts/perf_evidence.sh",
    "scripts/run_tests.sh",
    "scripts/ab.py",
    "scripts/bench_tables.py",
    "bench/scenes.hpp",
    "bench/search_bench.cpp",
    "bench/backends.hpp",
    "bench/reference.hpp",
    "support/scene.hpp",
)
DOC_ALLOWED = ("README.md", "AGENTS.md", "CLAUDE.md")
DOC_SUFFIXES = (".md", ".rst", ".adoc", ".txt")
CODE_ROOTS = ("src/", "include/")
MAX_AGENT_FILES = 12
MAX_AGENT_LINES = 1000

C_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp", ".ispc", ".isph")
HASH_SUFFIXES = (".sh", ".py", ".cmake")

BAD_COMMENT = (
    (r"\b\d+(\.\d+)?\s?(ms|us|µs|ns)\b", "a timing"),
    (r"\b\d+(\.\d+)?\s?%", "a percentage"),
    (r"\d\s?->\s?\d", "a before/after number"),
    (
        r"\b(uniform|huge_domain|sparse|slab|clustered|droplets|thin_pipe"
        r"|two_scale|surface)[ _]\d+[kKmM]\b",
        "a benchmark scene",
    ),
    (r"\bmeasured\b", "a measurement"),
    (
        r"\b(previously|no longer|used to|anymore|formerly)\b"
        r"|\bwas (changed|replaced|removed|moved)\b",
        "history",
    ),
)
INTRINSICS = re.compile(
    r"\b_mm\d*_\w+|\b__m(128|256|512)\w*|\w*intrin\.h|\b_tzcnt_\w+|\b_pext_\w+"
)
BAD_SUBJECT = (
    (r"^(win|checkpoint)\b", "a win/checkpoint tag"),
    (r"\b\d+(\.\d+)?\s?(ms|us|ns)\b", "a timing"),
    (r"\b\d+(\.\d+)?\s?%", "a percentage"),
    (r"\d\s?->\s?\d", "a before/after number"),
    (r"\bratio\b|\b\d\.\d+x\b", "a ratio"),
)
ATTRIBUTION = re.compile(
    r"co-authored-by:.*(claude|anthropic|opencode|deepseek)"
    r"|generated (with|by).*(claude|opencode)|claude\.ai/code",
    re.IGNORECASE,
)
REQUIRED_BUNDLE_FILES = ("meta.txt", "topdown_l1.txt", "hotspots.txt")


def git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def is_agent():
    pid = os.getppid()
    while pid > 1:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return False
        if re.search(rb"(^|[/\0 ])(opencode|claude)(\0| |$)", cmdline):
            return True
        pid = int(re.search(r"^PPid:\s*(\d+)", status, re.MULTILINE)[1])
    return False


def log_root():
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    return common.parent / ".perf-log"


def changed_files(*diff_args):
    files = []
    for line in git("diff", "--name-status", "--no-renames", *diff_args).splitlines():
        status, path = line.split("\t", 1)
        files.append((status[0], path))
    return files


def added_lines(*diff_args):
    path = None
    number = 0
    for line in git("diff", "--unified=0", "--no-renames", *diff_args).splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            number = int(re.search(r"\+(\d+)", line)[1])
        elif line.startswith("+") and path:
            yield path, number, line[1:]
            number += 1


def comment_text(path, line):
    if path == "scripts/agent_check.py":
        return None
    if path.endswith(C_SUFFIXES):
        match = re.search(r"//(.*)|/\*(.*)", line)
        if match:
            return match[1] or match[2] or ""
        stripped = line.strip()
        return stripped[1:] if stripped.startswith("*") else None
    if path.endswith(HASH_SUFFIXES) or Path(path).name == "CMakeLists.txt":
        match = re.search(r"(?:^|\s)#(.*)", line)
        return match[1] if match else None
    return None


def check_docs(files):
    errors = []
    for status, path in files:
        if status != "A":
            continue
        name = Path(path).name
        if path in DOC_ALLOWED or name in ("CMakeLists.txt", "SKILL.md"):
            continue
        if path.startswith("docs/") or path.endswith(DOC_SUFFIXES):
            errors.append(f"{path}: do not add documents to git; use .perf-log/")
    return errors


def check_lines(lines):
    errors = []
    for path, number, line in lines:
        comment = comment_text(path, line)
        if comment:
            for pattern, what in BAD_COMMENT:
                if re.search(pattern, comment, re.IGNORECASE):
                    errors.append(f"{path}:{number}: comment contains {what}: {line.strip()}")
                    break
        if path.startswith(CODE_ROOTS) and path.endswith(C_SUFFIXES) and INTRINSICS.search(line):
            errors.append(f"{path}:{number}: raw intrinsic, use ISPC: {line.strip()}")
    return errors


def readme_prose(text):
    return re.sub(
        r"<!-- bench:([\w-]+):start -->.*?<!-- bench:\1:end -->", "", text, flags=re.DOTALL
    )


def check_agent_change(files, staged_lines):
    errors = []
    branch = git("symbolic-ref", "--short", "-q", "HEAD").strip() if head_exists() else ""
    if branch in ("main", "master"):
        errors.append(f"agents do not commit to {branch}; create a branch")
    for _, path in files:
        if path.startswith(PROTECTED) or path in PROTECTED:
            errors.append(f"{path}: protected, needs approval")
    if any(path == "README.md" for _, path in files):
        staged = subprocess.run(
            ["git", "show", ":README.md"], text=True, capture_output=True
        ).stdout
        head = subprocess.run(
            ["git", "show", "HEAD:README.md"], text=True, capture_output=True
        ).stdout
        if readme_prose(staged) != readme_prose(head):
            errors.append("README.md: agents may only change generated benchmark tables")
    if len(files) > MAX_AGENT_FILES or staged_lines > MAX_AGENT_LINES:
        errors.append(
            f"commit is too large ({len(files)} files, {staged_lines} added lines); "
            "split it into atomic commits"
        )
    return errors


def head_exists():
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD"], capture_output=True
        ).returncode
        == 0
    )


def check_subject(subject):
    errors = []
    if len(subject) > 72:
        errors.append(f"subject is longer than 72 characters: {subject}")
    for pattern, what in BAD_SUBJECT:
        if re.search(pattern, subject, re.IGNORECASE):
            errors.append(f"subject contains {what}; put numbers in the body: {subject}")
    return errors


def read_log():
    path = log_root() / "experiments.jsonl"
    if not path.exists():
        return [], []
    entries, errors = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append((number, json.loads(line)))
        except json.JSONDecodeError as error:
            errors.append(f"experiments.jsonl:{number}: invalid JSON: {error}")
    return entries, errors


def check_log():
    entries, errors = read_log()
    root = log_root() / "evidence"
    seen = set()
    for number, entry in entries:
        where = f"experiments.jsonl:{number}"
        missing = [
            key
            for key in ("id", "level", "hypothesis", "result", "mechanism", "dead_end", "evidence")
            if key not in entry
        ]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
            continue
        if entry["id"] in seen:
            errors.append(f"{where}: duplicate id {entry['id']}")
        seen.add(entry["id"])
        if entry["level"] not in ("algorithm", "layout", "parallel", "simd", "micro"):
            errors.append(f"{where}: unknown level {entry['level']}")
        if entry["result"] not in ("kept", "rejected", "open"):
            errors.append(f"{where}: unknown result {entry['result']}")
        if entry["result"] == "open":
            continue
        if entry["result"] == "rejected" and entry["dead_end"] not in ("local", "global"):
            errors.append(f"{where}: a rejection must say whether the loss is local or global")
        if len(str(entry["mechanism"])) < 40:
            errors.append(f"{where}: mechanism must explain the result from measured data")
        if len(entry["evidence"]) < 2:
            errors.append(f"{where}: needs before and after evidence bundles")
        for label in entry["evidence"]:
            bundle = root / entry["id"] / label
            for name in REQUIRED_BUNDLE_FILES:
                if not (bundle / name).is_file():
                    errors.append(f"{where}: missing evidence {bundle / name}")
    return errors


def check_evidence_trailer(message):
    matches = re.findall(r"^Evidence:\s*(.+)$", message, re.MULTILINE)
    if not matches:
        return ["commit changes src/ or include/ and needs an Evidence: trailer"]
    errors = check_log()
    value = matches[-1].strip()
    if re.match(r"none\s*-\s*\S", value):
        return errors
    entries, _ = read_log()
    if not any(entry.get("id") == value and entry.get("result") == "kept" for _, entry in entries):
        errors.append(f"Evidence: {value} is not a kept experiment in the log")
    return errors


def report(errors):
    for error in errors:
        print(f"agent_check: {error}", file=sys.stderr)
    return 1 if errors else 0


def pre_commit():
    files = changed_files("--cached")
    lines = list(added_lines("--cached"))
    errors = check_docs(files) + check_lines(lines)
    if is_agent():
        errors += check_agent_change(files, len(lines))
    return report(errors)


def commit_msg(path):
    text = Path(path).read_text()
    message = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    subject = message.strip().splitlines()[0] if message.strip() else ""
    errors = check_subject(subject)
    if ATTRIBUTION.search(message):
        errors.append("commit message contains AI attribution or a session link")
    if is_agent() and any(p.startswith(CODE_ROOTS) for _, p in changed_files("--cached")):
        errors += check_evidence_trailer(message)
    return report(errors)


def check_range(base):
    spec = f"{base}...HEAD"
    files = changed_files(spec)
    errors = check_docs(files) + check_lines(added_lines(spec))
    for commit in git("rev-list", f"{base}..HEAD").split():
        message = git("log", "-1", "--format=%B", commit)
        errors += [f"{commit[:9]}: {e}" for e in check_subject(message.splitlines()[0])]
        if ATTRIBUTION.search(message):
            errors.append(f"{commit[:9]}: AI attribution or a session link")
        touched = git("show", "--format=", "--name-only", commit).splitlines()
        if any(p.startswith(CODE_ROOTS) for p in touched) and not re.search(
            r"^Evidence:\s*\S", message, re.MULTILINE
        ):
            print(f"agent_check: review {commit[:9]}, it changes code without an Evidence: trailer")
    protected =[p for _, p in files if p.startswith(PROTECTED) or p in PROTECTED]
    if protected:
        print("agent_check: review these protected files:\n  " + "\n  ".join(protected))
    if any(p == "README.md" for _, p in files):
        old = subprocess.run(["git", "show", f"{base}:README.md"], text=True, capture_output=True)
        if readme_prose(old.stdout) != readme_prose(Path("README.md").read_text()):
            print("agent_check: review README.md, text outside the benchmark tables changed")
    return report(errors)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "pre-commit":
        return pre_commit()
    if command == "commit-msg" and len(sys.argv) == 3:
        return commit_msg(sys.argv[2])
    if command == "range" and len(sys.argv) == 3:
        return check_range(sys.argv[2])
    if command == "log":
        return report(check_log())
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
