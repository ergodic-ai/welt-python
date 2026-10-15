"""API-122 / DEC-073: Graphite presentation, reading components and site outputs."""
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

import pytest

from scripts import build_docs

ROOT = build_docs.ROOT
ASSETS = ROOT / 'docs/assets'


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


@pytest.fixture(scope='module')
def site():
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(sys, 'argv', ['build_docs', '--output', folder]):
            build_docs.main()
        version = json.loads(next(Path(folder).glob('v*/manifest.json')).read_text())['sdk_version']
        yield Path(folder), Path(folder) / f'v{version}', version


def test_design_assets_match_recorded_provenance():
    provenance = json.loads((ASSETS / 'design-provenance.json').read_text())
    files = {entry['file'] for entry in provenance['assets']}
    assert 'tokens.css' in files and {f'fonts/{path.name}' for path in (ASSETS / 'fonts').glob('*.woff2')} <= files
    for entry in provenance['assets']:
        assert sha256((ASSETS / entry['file']).read_bytes()).hexdigest() == entry['sha256'], entry['file']
    assert (ASSETS / 'fonts/OFL-Geist.txt').read_text().count('SIL Open Font License') >= 1
    assert (ASSETS / 'fonts/OFL-Geist-Mono.txt').read_text().count('SIL Open Font License') >= 1


def test_stylesheet_takes_every_colour_from_tokens():
    css = (ASSETS / 'docs.css').read_text()
    # Only the declared docs-only token blocks and data-URI icon masks may hold literal colours.
    stripped = re.sub(r':root\s*\{[^}]*\}', '', css)
    stripped = re.sub(r'@media \(prefers-color-scheme: dark\) \{ :root:not\(\[data-theme="light"\]\) \{[^}]*\} \}', '', stripped)
    stripped = re.sub(r':root\[data-theme="dark"\] \{[^}]*\}', '', stripped)
    assert not re.findall(r'#[0-9a-fA-F]{3,8}\b|rgba?\(', stripped)
    tokens = (ASSETS / 'tokens.css').read_text()
    for name in set(re.findall(r'var\((--[\w-]+)\)', css)):
        assert name in tokens or f'{name}:' in css, name


def test_pages_make_no_third_party_requests_and_follow_the_system_theme(site):
    base, out, _ = site
    for path in [*out.glob('*.html'), base / 'index.html']:
        text = path.read_text()
        assert '<html lang="en">' in text, path.name
        for tag, attrs in Tags(text).tags:
            source = attrs.get('src') if tag in ('script', 'img') else attrs.get('href') if tag == 'link' and attrs.get('rel') != 'canonical' else None
            if source:
                assert not re.match(r'(https?:)?//', source), (path.name, source)
        head = text.split('</head>', 1)[0]
        assert "localStorage.getItem('welt-docs-theme')" in head and "t==='light'||t==='dark'" in head
    for path in out.glob('assets/*.css'):
        css = path.read_text()
        assert '@import' not in css and not re.search(r'url\((?!["\']?data:)(["\']?)(https?:)?//', css), path.name


def test_fences_render_tabs_files_highlights_and_outputs():
    text = ('# Guide\n\nIntro.\n\n```sh tab="pip"\npip install welt-client\n```\n\n```sh tab="uv"\nuv pip install welt-client\n```\n\n'
            '```python file="first.py" hl="2"\nfrom welt import Client\nclient = Client()\n```\n\n```output\n(150, 5)\n```\n')
    body, _ = build_docs.render_markdown(text)
    tags = Tags(body).tags
    tabs = [attrs for tag, attrs in tags if attrs.get('role') == 'tab']
    panels = {attrs['id']: attrs for tag, attrs in tags if attrs.get('role') == 'tabpanel'}
    assert [tab['aria-selected'] for tab in tabs] == ['true', 'false']
    for tab in tabs:
        panel = panels[tab['aria-controls']]
        assert panel['aria-labelledby'] == tab['id'] and ('hidden' in panel) == (tab['aria-selected'] == 'false')
    assert body.count('class="code-wrap') == 2
    assert 'code-file">first.py</span>' in body
    assert '<span class="hl">client = <span class="fn">Client</span>()</span>' in body
    assert '<div class="output"><div class="output-head">Output</div>' in body and '(150, 5)' in body
    assert '<p class="lead">Intro.</p>' in body


def test_long_opening_paragraphs_stay_body_text():
    body, _ = build_docs.render_markdown('# Guide\n\n' + 'Word ' * 90 + '\n')
    assert 'class="lead"' not in body


def test_callouts_steps_and_details():
    text = ('# Guide\n\n!!! tip "Why the name?"\n    Use `target`.\n\n!!! need "You\'ll need"\n    - Python 3.11\n\n'
            '## Install {.step}\n\nRun it.\n\n## Connect {.step}\n\nApprove.\n\n## Next steps\n\n'
            '<details markdown="1"><summary>Where is access stored?</summary>\n\nIn this process.\n\n</details>\n')
    body, _ = build_docs.render_markdown(text)
    assert '<div class="callout tip" role="note">' in body and '<p class="callout-title">Why the name?</p>' in body
    assert '<div class="callout need" role="note">' in body
    steps = body.split('<div class="steps">', 1)[1].split('</div>', 1)[0]
    assert steps.count('<section class="step">') == 2 and 'id="install"' in steps and 'id="connect"' in steps
    assert steps.count('class="step"') == 2  # only the sections, so each step is numbered once
    assert 'id="next-steps"' not in steps and body.index('</div>') < body.index('id="next-steps"')
    assert '<summary>Where is access stored?</summary>' in body and '<p>In this process.</p>' in body


@pytest.mark.parametrize('text, message', [
    ('!!! danger\n    Nope.\n', 'Unsupported callout type'),
    ('!!! note inline\n    Nope.\n', 'Unsupported callout type'),
    ('!!! tip-x\n    Nope.\n', 'Unsupported callout type'),
    ('```python file=first.py\nx = 1\n```\n', 'Malformed code fence attributes'),
    ('```python hl="3"\nx = 1\n```\n', 'Invalid highlighted line range'),
    ('```python hl="2-1"\nx = 1\ny = 2\n```\n', 'Invalid highlighted line range'),
    ('1. Step\n\n   ```sh\n   welt login\n   ```\n', 'multiple of four spaces'),
    ('!!! note\n    ## Inside {.step}\n\n    Text.\n', 'Step headings must be top-level'),
    ('```python colour="red"\nx = 1\n```\n', 'Unsupported code fence attribute'),
    ('```python hl="a-b"\nx = 1\n```\n', 'Invalid highlighted line range'),
])
def test_unsupported_components_refuse_publication(text, message):
    with pytest.raises(SystemExit, match=message):
        build_docs.render_markdown(text)


def test_code_stays_literal_inside_callouts_and_lists():
    text = '!!! note\n    ```python\n    x = "<script>alert(1)</script>"\n    ```\n\n- Step\n\n    ```sh\n    welt login\n    ```\n'
    body, _ = build_docs.render_markdown(text)
    assert '<script>' not in body and '&lt;script&gt;' in body
    assert 'WELTCODEBLOCK' not in body and body.count('class="code-wrap') == 2


def test_every_documented_constructor_parameter_is_described():
    reference = build_docs.reference()
    for name in build_docs.REQUIRED:
        section = reference.split(f'\n\n## {name}\n\n', 1)[1].split('\n\n### ', 1)[0]
        assert '| Parameter | Default | Description |' in section, name
    with patch.dict(build_docs.PARAMS, {'Classifier': {k: v for k, v in build_docs.PARAMS['Classifier'].items() if k != 'timeout'}}):
        with pytest.raises(SystemExit, match="Classifier need descriptions: missing \\['timeout'\\]"):
            build_docs.reference()


def test_long_signatures_wrap_one_parameter_per_line():
    reference = build_docs.reference()
    assert 'Client(\n    base_url=None,\n    api_key=None,\n    *,\n    timeout=60,' in reference
    assert 'Classifier.score(X, y, sample_weight=None)' in reference
    assert '**Returns:** ' in reference and '{: .contract }' in reference


def test_reference_index_links_only_existing_anchors(site):
    _, out, _ = site
    page = (out / 'reference.html').read_text()
    ids = set(re.findall(r' id="([^"]+)"', page))
    sidebar = page.split('<nav class="sidebar"', 1)[1].split('</nav>', 1)[0]
    anchors = re.findall(r'href="#([^"]+)"', sidebar)
    assert len(anchors) == len(build_docs.REQUIRED) + 2 and set(anchors) <= ids


def test_guides_have_pager_markdown_source_edit_link_and_llms_index(site):
    _, out, version = site
    llms = (out / 'llms.txt').read_text()
    assert 'https://welt.ergodic.dev/llms.txt' in llms
    order = [page for _, pages in build_docs.guide_groups(version) for _, page in pages]
    for index, name in enumerate(order):
        page = (out / f'{name}.html').read_text()
        assert (out / f'{name}.md').exists() and f'/v{version}/{name}.md' in llms
        assert f'data-markdown="{name}.md"' in page
        assert f'{build_docs.REPOSITORY}/issues/new?title=' in page and 'blob/main' not in page and 'tree/main' not in page
        pager = page.split('<nav class="pager"', 1)[1].split('</nav>', 1)[0]
        previous = order[index - 1] + '.html' if index else 'index.html'
        assert f'href="{previous}"' in pager
        if index + 1 < len(order):
            assert f'href="{order[index + 1]}.html"' in pager
    for path in out.glob('*.md'):
        assert '{{' not in path.read_text(), path.name


def test_models_table_renders_only_available_snapshot_profiles(site):
    _, out, _ = site
    snapshot = json.loads(build_docs.MODELS_SNAPSHOT.read_text())
    available = {(m['id'], t['task']) for m in snapshot['models'] for t in m['tasks'] if t['status'] == 'available' and t['execution_available']}
    planned = {m['id'] for m in snapshot['models'] if not any(t['status'] == 'available' for t in m['tasks'])}
    for page in ('capabilities.html', 'index.html'):
        table = (out / page).read_text().split('aria-label="Available models"', 1)[1].split('</table>', 1)[0]
        assert table.count('<tr>') == len(available) + 1
        for model in planned:
            assert f'<code>{model}</code>' not in table
    assert 'Public catalogue snapshot from ' + snapshot['taken_at'][:10] in (out / 'capabilities.html').read_text()
    with patch.object(build_docs, 'MODELS_SNAPSHOT', ROOT / 'missing.json'), pytest.raises(FileNotFoundError):
        build_docs.models_table()


def test_snapshot_keeps_only_rendered_catalogue_fields():
    from scripts import snapshot_models
    catalogue = {'items': [{'id': 'm', 'name': 'M', 'family': 'tabular', 'version': 'private-build', 'reason': None,
                            'task_profiles': [{'task': 'classification', 'status': 'available', 'execution_available': True,
                                               'version': 'v', 'limits': {'max_rows': 500, 'internal': 1}}]}]}
    reduced = snapshot_models.reduce(catalogue, 'https://example.invalid/v1/models', '2026-10-13T00:00:00Z')
    assert reduced['models'] == [{'id': 'm', 'name': 'M', 'tasks': [
        {'task': 'classification', 'status': 'available', 'execution_available': True, 'limits': {'max_rows': 500}}]}]


def test_navigation_covers_every_guide_once(site):
    _, out, _ = site
    pages = {path.stem for path in (ROOT / 'docs').glob('*.md')} | {'reference'}
    listed = [page for _, group in build_docs.GUIDES for _, page in group]
    assert sorted(listed) == sorted(set(listed)) and pages == set(listed)
    for label, page in build_docs.NAV:
        assert (out / f'{page}.html').exists(), label


def test_next_step_cards_render_links_from_markdown():
    body, _ = build_docs.render_markdown('## Next steps\n\n<div class="cards" markdown="1">\n\n- [Reuse a predictor](reuse.md): predict later.\n\n</div>\n')
    assert '<div class="cards">' in body and '<a href="reuse.html">Reuse a predictor</a> predict later.' in body


def test_placeholders_render_valid_blocks_in_every_position():
    cases = ['Run this:\n```sh\nwelt login\n```\nThen continue.\n',
             'Intro.\n\n    ```sh\n    welt login\n    ```\n',
             '1. Step\n\n    ```sh\n    welt login\n    ```\n\n2. Next\n']
    for text in cases:
        body, _ = build_docs.render_markdown(text)
        assert 'WELTCODE' not in body and body.count('class="code-wrap') == 1, text
        assert not re.search(r'<p>[^<]*<div', body) and '<pre><code><div' not in body, body
    numbered, _ = build_docs.render_markdown(cases[2])
    assert numbered.count('<ol>') == 1
    # Text that looks like an old placeholder stays literal.
    body, _ = build_docs.render_markdown('WELTCODEBLOCK0X\n\n```sh\nWELTCODEBLOCK0X\n```\n')
    assert body.count('WELTCODEBLOCK0X') == 2


def test_identical_tab_groups_get_distinct_ids_and_shell_highlights_nest():
    group = '```sh tab="pip"\npip install a\n```\n\n```sh tab="uv"\nuv pip install a\n```\n\nText.\n\n'
    body, _ = build_docs.render_markdown(group + group)
    ids = re.findall(r' id="(tabs-[^"]+)"', body)
    assert len(ids) == 8 and len(set(ids)) == 8
    body, _ = build_docs.render_markdown('```sh hl="2"\n# comment\nwelt login\n```\n')
    assert '<span class="co"># comment</span>\n<span class="hl">welt login</span>' in body


def test_tilde_fences_and_default_callout_labels():
    body, _ = build_docs.render_markdown('~~~sh\nwelt login\n~~~\n\n!!! need\n    - Python\n')
    assert 'class="code-wrap' in body and "<p class=\"callout-title\">You'll need</p>" in body.replace('&#x27;', "'")


def test_reference_escapes_raw_angle_text_and_markdown_copy_is_clean(site):
    _, out, _ = site
    assert '&lt;path&gt;.metadata.json' in (out / 'reference.html').read_text()
    assert '{: .contract }' not in (out / 'reference.md').read_text()


def test_models_table_refuses_a_non_public_snapshot(tmp_path):
    snapshot = json.loads(build_docs.MODELS_SNAPSHOT.read_text())
    snapshot['source'] = 'https://staging.invalid/v1/models'
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(snapshot))
    with patch.object(build_docs, 'MODELS_SNAPSHOT', path), pytest.raises(SystemExit, match='only a snapshot of'):
        build_docs.models_table()


def test_wrapped_signatures_keep_positional_only_and_keyword_only_markers():
    import inspect
    def sample(first_positional_argument, second_positional_argument, /, normal_argument=1, *, keyword_argument=2): pass
    text = build_docs.format_signature('Sample.method', inspect.signature(sample))
    assert text.split('\n')[1:-1] == ['    first_positional_argument,', '    second_positional_argument,', '    /,',
                                       '    normal_argument=1,', '    *,', '    keyword_argument=2,']


def test_card_descriptions_drop_the_markdown_separator_and_deep_fences_stay_valid():
    body, _ = build_docs.render_markdown('<div class="cards" markdown="1">\n\n- [Reuse a predictor](reuse.md): predict later.\n\n</div>\n\nText: kept.\n')
    assert '<a href="reuse.html">Reuse a predictor</a> predict later.' in body and 'Text: kept.' in body
    body, _ = build_docs.render_markdown('Intro.\n\n        ```sh\n        welt login\n        ```\n')
    assert body.count('class="code-wrap') == 1 and '<pre><code><div' not in body and '<pre><code>    <div' not in body


def test_product_switcher_and_ergodic_cross_links(site):
    """API-145 / DEC-078: the Products switcher, the ergodic continuation card and compatibility table."""
    base, out, version = site
    for path in [*out.glob('*.html'), base / 'index.html']:
        text = path.read_text()
        switcher = text.split('<details class="product-switcher">', 1)[1].split('</details>', 1)[0]
        assert '<span>Products</span>' in switcher and 'href="https://docs.ergodic.dev"' in switcher, path.name
        assert switcher.count('aria-current="true"') == 1 and 'assets/site.css' in text, path.name
    css = (ASSETS / 'site.css').read_text()
    assert not re.findall(r'#[0-9a-fA-F]{3,8}\b|rgba?\(', css)
    tokens = (ASSETS / 'tokens.css').read_text()
    assert all(name in tokens for name in set(re.findall(r'var\((--[\w-]+)\)', css)))
    # docs.css stays byte-identical to the copy the ergodic docs vendor; the switcher lives in site.css.
    assert '.product-switcher' not in (ASSETS / 'docs.css').read_text()
    guide = (out / 'causal-discovery.html').read_text()
    cards = guide.split('<div class="cards">', 1)[1].split('</div>', 1)[0]
    assert '<a href="https://docs.ergodic.dev/welt.html">Discover in Welt, estimate in ergodic</a>' in cards
    capabilities = (out / 'capabilities.html').read_text()
    assert 'id="compatibility-with-ergodic"' in capabilities and 'href="capabilities.html#compatibility-with-ergodic"' in guide
    assert '<code>ergodic.graph/1</code>' in capabilities and 'https://docs.ergodic.dev/compatibility.html' in capabilities
    assert 'https://docs.ergodic.dev/llms.txt' in (out / 'llms.txt').read_text()
