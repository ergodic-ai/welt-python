"""Check that guide rewrites keep executable code and published anchors (API-123 / DEC-073).

Compares docs/*.md against a base Git revision: every python/sh fence keeps its exact
content and order, and every heading ID rendered at the base still renders now.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docs import ROOT, render_markdown  # noqa: E402

FENCE = re.compile(r'^[ \t]*(`{3,})(python|sh)[ \t]*\n(.*?)\n[ \t]*\1[ \t]*$', re.M | re.S)


def code(text):
    return [(language, body) for _, language, body in FENCE.findall(text)]


def ids(text):
    return re.findall(r'<h[1-6] id="([^"]+)"', render_markdown(text)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='fcf7313', help='Git revision before the rewrite')
    args = parser.parse_args()
    problems = []
    for path in sorted((ROOT / 'docs').glob('*.md')):
        relative = path.relative_to(ROOT).as_posix()
        try:
            before = subprocess.check_output(['git', 'show', f'{args.base}:{relative}'], cwd=ROOT, text=True)
        except subprocess.CalledProcessError:
            continue
        after = path.read_text()
        if code(before) != code(after):
            old, new = code(before), code(after)
            first = next((i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]), min(len(old), len(new)))
            problems.append(f'{relative}: python/sh fences changed (counts {len(old)} → {len(new)}, first difference at fence {first + 1})')
        lost = sorted(set(ids(before)) - set(ids(after)))
        if lost:
            problems.append(f'{relative}: lost heading anchors {lost}')
    for problem in problems:
        print(problem)
    print('OK: code fences and anchors preserved' if not problems else f'{len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
