"""Regressions for current reference contracts, precise search and old deep links."""
import json
import re
import shutil
import subprocess
from unittest import TestCase, skipUnless

from scripts import build_docs


class AuditReference(TestCase):
    def test_supported_inherited_scores_and_batch_returns_are_documented(self):
        reference = build_docs.reference()
        for estimator in ('Classifier', 'Regressor'):
            self.assertIn(f'### {estimator}.score', reference)
            self.assertIn(f'{estimator}.score(X, y, sample_weight=None)', reference)
        self.assertNotIn('### Classifier.set_score_request', reference)
        self.assertNotIn('### Regressor.set_score_request', reference)
        for job in ('Job', 'AsyncJob'):
            section = reference.split(f'### {job}.result\n', 1)[1].split('\n### ', 1)[0].split('\n## ', 1)[0]
            for field in ('partially_completed', 'row_count', 'successful_ranges', 'failed_ranges',
                          'result_reference', 'expires_at', 'batch_payload', 'null predictions'):
                self.assertIn(field, section)
            self.assertIn('[start, stop)', section)

    def test_usage_and_credit_recovery_match_current_wire_fields(self):
        reference = build_docs.reference()
        self.assertNotIn('No rates, credits, pricing', reference)
        for client in ('Client', 'AsyncClient'):
            section = reference.split(f'### {client}.usage\n', 1)[1].split('\n### ', 1)[0].split('\n## ', 1)[0]
            for term in ('nullable credits', 'rate_card_version', 'reservation', 'settlement',
                         'null means unpriced', 'differs from zero', 'no raw prediction payloads'):
                self.assertIn(term, section)
        for term in ('WeltError (HTTP 402)', 'insufficient_credits', 'spend_cap_reached',
                     'plan_past_due', 'Workspace Billing', 'pending reservations'):
            self.assertIn(term, reference)

    def test_method_search_uses_the_real_rendered_anchors(self):
        body, _ = build_docs.render_markdown(build_docs.reference())
        records = build_docs.search_records(body, 'reference', 'Python reference')
        ids = set(re.findall(r' id="([^"]+)"', body))
        methods = {record['method']: record for record in records if 'method' in record}
        for name, anchor in (('Client.submit_model_run', 'clientsubmit_model_run'),
                             ('AsyncClient.submit_model_run', 'asyncclientsubmit_model_run'),
                             ('Classifier.score', 'classifierscore'),
                             ('Regressor.score', 'regressorscore'),
                             ('Job.result', 'jobresult'), ('AsyncJob.result', 'asyncjobresult')):
            self.assertEqual(methods[name]['href'], 'reference.html#' + anchor)
            self.assertIn(anchor, ids)
        # H2 class summaries do not absorb all method bodies and displace a precise match.
        client = next(record for record in records if record['href'] == 'reference.html#client')
        self.assertNotIn('submit_model_run', client['text'])

    def test_search_preserves_renderer_ids_for_repeated_and_formatted_titles(self):
        body, _ = build_docs.render_markdown('# Guide\n\n## [A title](start.md)\n\nFirst.\n\n## A title\n\nSecond.')
        records = build_docs.search_records(body, 'guide', 'Guide')
        self.assertEqual([record['href'] for record in records],
                         ['guide.html', 'guide.html#a-title', 'guide.html#a-title_1'])


@skipUnless(shutil.which('node'), 'Node is required to execute documentation JavaScript contracts')
class AuditJavaScript(TestCase):
    def javascript(self, code):
        result = subprocess.run(['node', '-e', code], text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_actual_redirect_script_preserves_query_fragment_and_same_origin(self):
        page = build_docs.current_redirect('reference.html')
        self.assertIn('<a id="current-docs" href="../reference.html">', page)
        self.assertNotIn('http-equiv="refresh"', page)
        script = re.search(r'<script>(.*?)</script>', page, re.S)[1]
        origin = 'https://ergodic-ai.github.io/welt-python/v0.8.0/reference.html'
        cases = [('?source=old%20guide&next=https%3A%2F%2Fexample.test', '#clientpredict'),
                 ('?download=1', '#regression-predict-a-number'), ('', '')]
        for query, fragment in cases:
            source = (f'const location = {{href: {json.dumps(origin + query + fragment)}, '
                      f'search: {json.dumps(query)}, hash: {json.dumps(fragment)}, replace: (value) => location.redirected = value}};'
                      'const window = {location}; const link = {}; const document = {getElementById: () => link};'
                      + script + '\nconsole.log(JSON.stringify({redirected: location.redirected, fallback: link.href}));')
            result = self.javascript(source)
            expected = 'https://ergodic-ai.github.io/welt-python/reference.html' + query + fragment
            self.assertEqual(result, {'redirected': expected, 'fallback': expected})

    def test_real_search_ranking_prefers_exact_sync_async_and_inherited_methods(self):
        js = (build_docs.ROOT/'docs/assets/docs.js').read_text()
        rank = re.search(r'const searchRank = \(item, query\) => \{.*?\n  \};', js, re.S)[0]
        body, _ = build_docs.render_markdown(build_docs.reference())
        records = build_docs.search_records(body, 'reference', 'Python reference')
        # A class-level match is deliberately placed before the exact method.
        records.insert(0, dict(title='Python reference · Client', text='Client.submit_model_run', href='reference.html#client'))
        for method in ('Client.submit_model_run', 'AsyncClient.submit_model_run', 'Classifier.score', 'Regressor.score'):
            source = (rank + f'\nconst index = {json.dumps(records)}; const query = {json.dumps(method.lower())};'
                      'const words = query.split(/\\s+/);'
                      'const found = index.filter(item => words.every(word => `${item.title} ${item.text}`.toLowerCase().includes(word)))'
                      '.sort((a, b) => searchRank(b, query) - searchRank(a, query));'
                      'console.log(JSON.stringify(found[0]));')
            result = self.javascript(source)
            self.assertEqual(result['method'], method)
            self.assertEqual(result['href'], 'reference.html#' + method.lower().replace('.', ''))
