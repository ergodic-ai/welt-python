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
    assert 'download' not in body.lower()


def test_notebook_has_prepared_df_real_rows_holdout_and_observed_report():
    body=current_home();parsed=NotebookPage();parsed.feed(body)
    assert 'already-loaded pandas DataFrame' in body
    assert 'These cells do not load data' in body
    assert 'classes 0–2' in body
    assert 'df.head()\n' in parsed.code
    assert 'aria-label="First five prepared Iris rows"' in body
    assert '<td>1.0</td><td>0.2</td><td>4.6</td><td>3.6</td>' in body
    program=next(code for code in parsed.code if 'train, test = train_test_split' in code)
    compile(program,'visible notebook cell','exec')
    assert 'random_state=9, stratify=df["label"]' in program
    assert 'X_train, y_train = train.drop(columns="label"), train["label"]' in program
    assert 'X_test, y_test = test.drop(columns="label"), test["label"]' in program
    assert 'model.fit(X_train, y_train)' in program and 'model.predict(X_test)' in program
    assert 'print(classification_report(y_test, predictions, zero_division=0))' in program
    assert 'accuracy                           1.00        30' in body
    assert 'one observed result, not a promised score' in body
    assert body.index('df.head()') < body.index('First five prepared Iris rows') < body.index('train, test') < body.index('Recorded classification report')


def test_current_theme_rules_do_not_restylize_old_homepage():
    css=(Path(__file__).resolve().parents[1]/'docs/assets/docs.css').read_text()
    current=css.split('/* API-075:',1)[1]
    assert '.home:has(.home-v080)' in current
    assert ':root[data-theme=light] .home:has(.home-v080)' in current
    assert '--panel:#fff' in current and '--panel:#181a1c' in current
    assert '--paper' not in current and 'color-scheme:light' not in current
