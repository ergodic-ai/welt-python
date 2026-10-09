"""API-072: make the real training DataFrame visible without hiding setup in Copy."""
from html.parser import HTMLParser
from pathlib import Path

from scripts.build_docs import home_body


class Examples(HTMLParser):
    def __init__(self):
        super().__init__();self.copied=[]
    def handle_starttag(self, tag, attributes):
        attributes=dict(attributes)
        if 'data-copy' in attributes:self.copied.append(attributes['data-copy'])


def test_title_then_three_steps_no_animation_or_marketing():
    start=(Path(__file__).resolve().parents[1]/'docs/start.md').read_text()
    body=home_body(start,sdk_version='0.7.1')
    assert 'From a DataFrame' in body
    assert body.index('1 &nbsp; Install') < body.index('2 &nbsp; Connect') < body.index('3 &nbsp; Train')
    for removed in ('class="visual"','id="motion"','Your table. A foundation model.','Start with Python you already know.'):
        assert removed not in body
    assert 'Client().connect()' not in body
    parsed=Examples();parsed.feed(body)
    assert len(parsed.copied)==2
    for program in parsed.copied:
        compile(program,'complete copied example','exec')
        assert 'split = train_test_split(' in program
        assert 'train = split[0]' in program and 'print(train.head())' in program
        assert 'test = split[1]' in program and 'test.drop(columns=target)' in program
        assert 'download_research_dataset' in program
