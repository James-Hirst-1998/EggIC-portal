// Test results page: the same logins as the test, showing fixed totals from
// /test-results.json (the three full testers and the live AI model, 2026-11-19)
// and a few photos, signed like the test's own.
import { signIn, signOut, onUserChange } from './services/auth.js';
import { listTestImages, signTestImages } from './services/testImages.js';
import { SPECIES } from './species.js';

const $ = id => document.getElementById(id);
const track = (name, props) => window.track?.(name, props);
const NAME = Object.fromEntries(SPECIES.map(s => [s.key, s.name]));
const pct = (n, d) => `${Math.round((100 * n) / d)}%`;
const WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'];
const LIVE = 95;   // CONFIDENT_AT in modal/app.py, as a percentage

function show(id) {
  for (const s of document.querySelectorAll('main > section')) s.classList.toggle('show', s.id === id);
}

function message(title, text) {
  $('message-title').textContent = title;
  $('message-text').textContent = text;
  show('message');
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

/* ---------- theme, shared with the identifier ---------- */
if (localStorage.getItem('eggic-theme') === 'dark') document.documentElement.dataset.theme = 'dark';
$('theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  if (dark) delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = 'dark';
  localStorage.setItem('eggic-theme', dark ? 'light' : 'dark');
});

$('zoom').addEventListener('click', () => $('zoom').close());

/* ---------- login ---------- */
$('login-form').addEventListener('submit', async e => {
  e.preventDefault();
  $('login-btn').disabled = true;
  $('login-error').textContent = '';
  try {
    await signIn($('email').value.trim(), $('password').value);
  } catch (err) {
    console.error(err);
    $('login-error').textContent = 'That email and password did not match.';
  } finally {
    $('login-btn').disabled = false;
  }
});

$('signout').addEventListener('click', () => signOut().then(() => window.forget?.()).catch(console.error));

onUserChange(user => {
  $('signout').style.display = user ? '' : 'none';
  if (user) window.identify?.(user.id, { email: user.email });
  if (user) start().catch(err => message('Something went wrong', err.message));
  else show('login');
});

/* ---------- the results ---------- */
async function start() {
  show('loading');
  const res = await fetch('/test-results.json');
  if (!res.ok) throw new Error(`Could not load the results (HTTP ${res.status}).`);
  const [data, names] = await Promise.all([res.json(), listTestImages()]);
  if (!names.length) {
    return message('No results for this login', 'Ask James to check your account is set up for the test.');
  }
  const urls = await signTestImages(data.examples.map(x => x.image_path));
  render(data, urls);
  track('results_viewed', { total: names.length });
  show('results');
}

function render(d, urls) {
  const p = d.people;
  $('lede').textContent =
    `${WORDS[p.count] ?? p.count} Shark Trust staff each looked at the same ${d.photos} egg case photos and ` +
    `named the species. The AI model was given the same ${d.photos}. Every photo has a known ` +
    'answer, and both were marked against it.';

  $('headline').replaceChildren(
    tile('People', `${pct(p.right, p.answers)} right`,
      `${p.right} of ${p.answers} answers. Each person got between ${p.min} and ${p.max} of ${d.photos}.`),
    tile('AI model', `${pct(d.model.right, d.photos)} right`,
      `${d.model.right} of ${d.photos}, giving its best guess on every photo.`),
  );

  $('limits').replaceChildren(...d.lines.map(l => {
    const at = Math.round(l.at * 100);
    return tile(`Confidence limit ${at}%${at === LIVE ? ', as on the live identifier' : ''}`,
      `${pct(l.right, l.answered)} right`,
      `Names ${l.answered} of ${d.photos} photos (${pct(l.answered, d.photos)}) and gets ` +
      `${l.right === l.answered ? 'every one' : `${l.right} of them`} right. ` +
      `Says "not sure" on the other ${d.photos - l.answered}.`);
  }));

  const rate = s => s.people_right / s.people_answers;
  const species = [...d.species].sort((a, b) => rate(a) - rate(b));
  $('species').replaceChildren(
    row('th', ['Species', 'Photos', 'People right', 'AI model right']),
    ...species.map(s => {
      const tr = row('td', [NAME[s.key], s.photos,
        `${pct(s.people_right, s.people_answers)} (${s.people_right} of ${s.people_answers})`,
        `${pct(s.model_right, s.photos)} (${s.model_right} of ${s.photos})`]);
      if (rate(s) < 0.85) tr.children[2].classList.add('low');
      if (s.model_right / s.photos < 0.85) tr.children[3].classList.add('low');
      return tr;
    }),
  );
  $('mixups').textContent = 'The mix-ups people made most: ' + d.mixups
    .map(m => `${NAME[m.a]} and ${NAME[m.b]} (${m.n} answers)`).join(', ') + '.';

  $('examples').replaceChildren(...d.examples.map(x => example(x, urls.get(x.image_path))));
}

function tile(k, v, s) {
  return el('div', { className: 'tile' },
    el('div', { className: 'k', textContent: k }),
    el('div', { className: 'v', textContent: v }),
    el('div', { className: 's', textContent: s }));
}

function row(cell, values) {
  return el('tr', {}, ...values.map((v, i) =>
    el(cell, { textContent: String(v), className: i ? 'n' : '' })));
}

function example(x, url) {
  const said = Object.entries(x.people.reduce((n, a) => ({ ...n, [a]: (n[a] || 0) + 1 }), {}))
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${n} said ${NAME[k] ?? 'not sure'}`).join(', ');
  const m = x.model;
  const call = `${NAME[m.top] ?? m.top}, ${Math.round(m.confidence * 100)}% confident`;
  const img = el('img', { src: url, alt: `Egg case: ${NAME[x.true]}`, loading: 'lazy' });
  img.addEventListener('click', () => {
    $('zoom-img').src = url;
    $('zoom-img').alt = img.alt;
    $('zoom').showModal();
  });
  return el('article', { className: 'ex' }, img,
    el('div', { className: 'body' },
      el('div', { className: 'tag', textContent: x.tag }),
      el('dl', {},
        el('dt', { textContent: 'Answer' }), el('dd', { textContent: NAME[x.true] }),
        el('dt', { textContent: 'People' }), el('dd', { textContent: said }),
        el('dt', { textContent: 'AI model' }), el('dd', { textContent: call })),
      el('p', { textContent: x.text })));
}
