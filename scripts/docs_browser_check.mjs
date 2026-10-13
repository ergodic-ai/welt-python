// API-122: drive a built docs site in headless Chrome over the DevTools protocol and print
// JSON results. Usage: node docs_browser_check.mjs <chrome> <base-url> <profile-dir>
import { spawn } from 'node:child_process';

const [,, chromePath, base, profile] = process.argv;
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(chromePath, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let targets;
for (let i = 0; i < 100 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(200); } }
const socket = new WebSocket(targets.find((target) => target.type === 'page').webSocketDebuggerUrl);
await new Promise((resolve) => socket.addEventListener('open', resolve));
let counter = 0; const pending = new Map(); const errors = [];
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id); }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((resolve) => { const id = ++counter; pending.set(id, resolve); socket.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => { const reply = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); return reply.result.exceptionDetails ? { error: reply.result.exceptionDetails.text } : reply.result.result.value; };
const key = async (name, code, keyCode, modifiers = 0, text) => { await send('Input.dispatchKeyEvent', { type: 'keyDown', key: name, code, windowsVirtualKeyCode: keyCode, modifiers, text }); await send('Input.dispatchKeyEvent', { type: 'keyUp', key: name, code, windowsVirtualKeyCode: keyCode, modifiers }); };
const go = async (path, width = 1440, dark = false) => {
  await send('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: width < 600 });
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' }] });
  await send('Page.navigate', { url: `${base}/${path}` });
  for (let i = 0; i < 50; i++) { await sleep(100); if (await evaluate("document.readyState === 'complete'")) break; }
  await sleep(150);
};
const results = {};
await send('Page.enable'); await send('Runtime.enable');

await go('start.html', 1440, true);
results.systemDark = await evaluate("!document.documentElement.dataset.theme && getComputedStyle(document.body).backgroundColor === 'rgb(9, 9, 11)'");
await evaluate("document.querySelector('#theme').click()");
results.choiceStored = await evaluate("localStorage.getItem('welt-docs-theme') === 'light'");
await go('reference.html', 1440, true);
results.choiceAppliesElsewhere = await evaluate("document.documentElement.dataset.theme === 'light' && getComputedStyle(document.body).backgroundColor === 'rgb(255, 255, 255)'");
await evaluate("localStorage.removeItem('welt-docs-theme')");

await go('start.html');
await key('k', 'KeyK', 75, 2);
await sleep(300);
results.ctrlKOpensPalette = await evaluate("document.querySelector('#search-dialog').open && document.activeElement.id === 'search-input'");
await send('Input.insertText', { text: 'jobtimeout' });
await sleep(400);
results.searchResults = await evaluate("document.querySelectorAll('#search-results [role=option]').length");
await key('ArrowDown', 'ArrowDown', 40);
results.arrowSelects = await evaluate("document.querySelector('#search-results [aria-selected=true]')?.id === document.querySelector('#search-input').getAttribute('aria-activedescendant')");
await key('Escape', 'Escape', 27);
await sleep(200);
results.escapeClosesAndRestoresFocus = await evaluate("!document.querySelector('#search-dialog').open && document.activeElement.id === 'search'");
await key('/', 'Slash', 191, 0, '/');
await sleep(300);
results.slashOpensPalette = await evaluate("document.querySelector('#search-dialog').open");
await key('Escape', 'Escape', 27);

await go('reference.html');
await key('k', 'KeyK', 75, 2);
await sleep(300);
await send('Input.insertText', { text: 'classifier.fit' });
await sleep(400);
await key('Enter', 'Enter', 13, 0, '\r');
await sleep(500);
results.samePageResultFocusesHeading = await evaluate("!document.querySelector('#search-dialog').open && document.activeElement.id === 'classifierfit'");
await evaluate("document.querySelector('#version').focus()");
await key('k', 'KeyK', 75, 2);
await sleep(300);
await key('Escape', 'Escape', 27);
await sleep(300);
results.escapeReturnsToPreviousFocus = await evaluate("document.activeElement.id === 'version'");

await go('index.html');
await evaluate("document.querySelector('[role=tab]').focus()");
await key('ArrowRight', 'ArrowRight', 39);
results.tabArrowKeys = await evaluate("document.activeElement.getAttribute('role') === 'tab' && document.activeElement.getAttribute('aria-selected') === 'true' && !document.getElementById(document.activeElement.getAttribute('aria-controls')).hidden");
results.focusVisible = await evaluate("getComputedStyle(document.activeElement).outlineStyle !== 'none'");

const pages = ['index.html', 'start.html', 'reference.html', 'capabilities.html', 'cli.html', 'research-datasets.html', 'notebook-classification.html'];
results.overflow = {};
for (const width of [320, 375, 1024]) {
  for (const page of pages) {
    await go(page, width);
    const wide = await evaluate('document.documentElement.scrollWidth - innerWidth');
    if (wide > 0) results.overflow[`${page}@${width}`] = wide;
  }
}
await go('start.html', 375);
await evaluate("document.querySelector('#menu').click()");
await sleep(200);
results.mobileDrawer = await evaluate("document.querySelector('#navigation').open && !!document.querySelector('#navigation [data-version]').offsetParent && !!document.querySelector('#navigation a[href=\"https://welt.ergodic.dev\"]').offsetParent");
await go('start.html', 1024);
results.pageActionsWithoutOutline = await evaluate("!!document.querySelector('.page-actions-inline [data-markdown]').offsetParent && !!document.querySelector('.page-actions-inline a[href*=\"issues/new\"]').offsetParent");
results.steps = await evaluate("document.querySelectorAll('.steps > section.step').length > 0 && document.querySelectorAll('h2.step').length === 0 && getComputedStyle(document.querySelector('section.step'), '::before').content !== 'none'");
results.errors = errors;
console.log(JSON.stringify(results));
socket.close(); chrome.kill();
process.exit(0);
