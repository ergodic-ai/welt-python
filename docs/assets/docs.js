/* Local-only documentation interactions. No telemetry, credentials or live execution. */
(() => {
  const root = document.documentElement;
  try { root.dataset.theme = localStorage.getItem('welt-docs-theme') || 'dark'; } catch (_) {}
  const theme = document.querySelector('#theme');
  const updateTheme = () => { theme.textContent = root.dataset.theme === 'light' ? 'Dark' : 'Light'; theme.setAttribute('aria-label', `Switch to ${root.dataset.theme === 'light' ? 'dark' : 'light'} theme`); };
  updateTheme();
  theme.addEventListener('click', () => { root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light'; try { localStorage.setItem('welt-docs-theme', root.dataset.theme); } catch (_) {} updateTheme(); });
  const announce = (value) => { const live = document.querySelector('#announcement'); live.textContent = ''; requestAnimationFrame(() => { live.textContent = value; }); };
  document.querySelectorAll('.code-wrap').forEach((wrap) => {
    const code = wrap.querySelector('code');
    const button = document.createElement('button'); button.className = 'copy'; button.type = 'button'; button.textContent = wrap.dataset.copyLabel || 'Copy'; button.setAttribute('aria-label', wrap.dataset.copyLabel || 'Copy code'); wrap.append(button);
    button.addEventListener('click', async () => { try { await navigator.clipboard.writeText(wrap.dataset.copy || code.textContent); button.textContent = 'Copied'; announce('Code copied to clipboard.'); setTimeout(() => { button.textContent = wrap.dataset.copyLabel || 'Copy'; }, 2200); } catch (_) { button.textContent = 'Select code'; announce('Clipboard unavailable. Select and copy the code below.'); if (wrap.dataset.copy) { let fallback = wrap.querySelector('.copy-fallback'); if (!fallback) { fallback = document.createElement('textarea'); fallback.className = 'copy-fallback'; fallback.readOnly = true; fallback.rows = 12; fallback.setAttribute('aria-label', 'Complete Python example to select and copy'); fallback.value = wrap.dataset.copy; wrap.append(fallback); } fallback.focus(); fallback.select(); announce('Clipboard unavailable. The complete example is selected below; copy it with your keyboard.'); } else { const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents(code); selection.removeAllRanges(); selection.addRange(range); } } });
  });
  document.querySelectorAll('[role=tablist]').forEach((list) => {
    const tabs = [...list.querySelectorAll('[role=tab]')];
    const select = (tab) => { tabs.forEach((item) => { const active = item === tab; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1; document.getElementById(item.getAttribute('aria-controls')).hidden = !active; }); };
    tabs.forEach((tab, i) => { tab.addEventListener('click', () => select(tab)); tab.addEventListener('keydown', (event) => { let next; if (event.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length]; if (event.key === 'ArrowLeft') next = tabs[(i + tabs.length - 1) % tabs.length]; if (event.key === 'Home') next = tabs[0]; if (event.key === 'End') next = tabs[tabs.length - 1]; if (next) { event.preventDefault(); select(next); next.focus(); } }); });
  });
  document.querySelector('#version').addEventListener('change', (event) => { location.href = event.target.value; });
  const openDialog = (id, opener) => { const dialog = document.getElementById(id); dialog.showModal(); const restore = () => { opener.focus(); dialog.removeEventListener('close', restore); }; dialog.addEventListener('close', restore); };
  document.querySelector('#menu').addEventListener('click', (event) => openDialog('navigation', event.currentTarget));
  document.querySelectorAll('dialog [data-close]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach((dialog) => dialog.addEventListener('keydown', (event) => { if (event.key === 'Escape') { event.preventDefault(); dialog.close(); } }));
  const searchButton = document.querySelector('#search'); const search = document.querySelector('#search-dialog');
  const searchURL = new URL(document.body.dataset.search, location.href);
  let index;
  const field = document.querySelector('#search-input'); const results = document.querySelector('#search-results'); const empty = document.querySelector('#search-empty');
  const render = () => { results.replaceChildren(); const words = field.value.toLowerCase().trim().split(/\s+/).filter(Boolean); const found = words.length && index ? index.filter((item) => words.every((word) => `${item.title} ${item.text}`.toLowerCase().includes(word))).slice(0, 12) : []; empty.hidden = found.length > 0; empty.textContent = !words.length ? 'Search this SDK version by task, method or error.' : !index ? 'Search is unavailable. Browse Guides or Reference in the navigation.' : 'No matches. Try “reopen”, “target”, “JobTimeoutError” or “causal”.'; found.forEach((item) => { const li = document.createElement('li'); const a = document.createElement('a'); a.href = new URL(item.href, searchURL).href; a.textContent = item.title; const small = document.createElement('small'); small.textContent = item.text.slice(0, 125); a.append(small); li.append(a); results.append(li); }); announce(found.length ? `${found.length} search results.` : empty.textContent); };
  searchButton.addEventListener('click', async () => { openDialog('search-dialog', searchButton); field.focus(); if (!index) { try { const response = await fetch(searchURL); if (!response.ok) throw Error(); index = await response.json(); } catch (_) { index = null; } } render(); });
  field.addEventListener('input', render);
  document.addEventListener('keydown', (event) => { if (event.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) && !search.open && !document.querySelector('dialog[open]')) { event.preventDefault(); searchButton.click(); } });
})();
