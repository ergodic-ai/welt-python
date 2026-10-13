"""API-072: make the real training DataFrame visible without hiding setup in Copy."""
from html.parser import HTMLParser
import re
import sys
from unittest.mock import patch

import pytest
from pathlib import Path

from scripts.build_docs import home_body


class Examples(HTMLParser):
    def __init__(self):
        super().__init__();self.copied=[]
    def handle_starttag(self, tag, attributes):
        attributes=dict(attributes)
        if 'data-copy' in attributes:self.copied.append(attributes['data-copy'])


def test_pre_080_landing_is_never_rederived_from_current_guides():
    """API-123: the 0.7.1 landing renders from its pinned baseline in sdk-docs.yml."""
    start=(Path(__file__).resolve().parents[1]/'docs/start.md').read_text()
    with pytest.raises(SystemExit, match='pinned presentation baseline'):
        home_body(start,sdk_version='0.7.1')
    workflow=(Path(__file__).resolve().parents[1]/'.github/workflows/sdk-docs.yml').read_text()
    assert '.presentation-baseline/scripts/build_historical_docs.py --source .legacy-sdk071 --version 0.7.1' in workflow
    from scripts import build_historical_docs
    with patch.object(sys,'argv',['historical','--source','.','--output','site','--version','0.7.1']):
        with pytest.raises(SystemExit, match='pinned presentation baseline'):
            build_historical_docs.main()


class NotebookPage(HTMLParser):
    def __init__(self):
        super().__init__();self.tabs=[];self.panels=[];self.code=[];self.in_code=False
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if attrs.get('role')=='tab':self.tabs.append(attrs)
        if attrs.get('role')=='tabpanel':self.panels.append(attrs)
        if tag=='code':self.in_code=True;self.code.append('')
    def handle_endtag(self,tag):
        if tag=='code':self.in_code=False
    def handle_data(self,data):
        if self.in_code:self.code[-1]+=data


def current_home():
    start=(Path(__file__).resolve().parents[1]/'docs/start.md').read_text()
    return home_body(start,sdk_version='0.8.0',browser_connect=True)


def test_current_three_cards_and_independent_accessible_tab_groups():
    body=current_home();parsed=NotebookPage();parsed.feed(body)
    assert body.count('class="start-card')==3
    assert 'aria-label="Install command"' in body and 'aria-label="Connection method"' in body
    assert len(parsed.tabs)==len(parsed.panels)==5
    assert len({tab['id'] for tab in parsed.tabs})==5
    panels={panel['id']:panel for panel in parsed.panels}
    for tab in parsed.tabs:
        panel=panels[tab['aria-controls']]
        assert panel['aria-labelledby']==tab['id']
        assert ('hidden' not in panel)==(tab['aria-selected']=='true')
        assert tab['tabindex']==('0' if tab['aria-selected']=='true' else '-1')
    assert "uv pip install 'welt-client[parquet]'\n" in parsed.code
    assert "pip install 'welt-client[parquet]'\n" in parsed.code
    assert 'welt login\n' in parsed.code
    assert 'from welt import Client\n\nClient().connect()\n' in parsed.code
    assert any('getpass("Welt API key: ")' in item for item in parsed.code)
    assert '==0.8' not in body and 'data-copy=' not in body


def test_notebook_loads_current_iris_before_holdout_and_observed_report():
    body=current_home();parsed=NotebookPage();parsed.feed(body)
    assert 'Load Iris below' in body
    assert 'Workspace → Billing' in body
    assert 'classes 0–2' in body
    loading=next(code for code in parsed.code if 'load_research_dataset' in code)
    compile(loading,'visible loading cell','exec')
    assert 'target = "target"' in loading and 'data.head()' in loading
    assert 'aria-label="First five current Iris rows"' in body
    assert '<td>4.6</td><td>3.6</td><td>1.0</td><td>0.2</td><td>0</td>' in body
    program=next(code for code in parsed.code if 'train, test = train_test_split' in code)
    compile(program,'visible notebook cell','exec')
    assert 'random_state=9, stratify=data[target]' in program
    assert 'X_train, y_train = train.drop(columns=target), train[target]' in program
    assert 'X_test, y_test = test.drop(columns=target), test[target]' in program
    assert 'model.fit(X_train, y_train)' in program and 'model.predict(X_test)' in program
    assert 'print(classification_report(y_test, predictions, zero_division=0))' in program
    assert 'accuracy                           1.00        30' in body
    assert 'one observed result, not a promised score' in body
    # Syntax highlighting adds markup inside code, so code order is checked on visible text.
    text=re.sub(r'<[^>]+>','',body)
    assert text.index('data.head()') < text.index('train, test = train_test_split')
    head=body.index('data.<span class="fn">head</span>()')
    assert head < body.index('First five current Iris rows') < body.index('train_test_split') < body.index('Recorded classification report')
