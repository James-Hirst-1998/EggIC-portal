// Staff test page: one photo at a time, nine species plus Not sure.
// This file owns the page; every Supabase call goes through ./services.
import { signIn, signOut, onUserChange } from './services/auth.js';
import { listTestImages, signTestImages, signTestImage } from './services/testImages.js';
import { myAnswers, saveAnswer } from './services/testAnswers.js';
import { SPECIES, NOT_SURE } from './species.js';
import { seededShuffle } from './lib/seededShuffle.js';
import { zoomable } from './ui/zoom.js';

const $ = id => document.getElementById(id);
const PRELOAD = 3;        // photos fetched ahead, so moving on is instant
const SETTLE_MS = 250;    // answers this soon after a photo appears are a double click

const state = {
  user: null,
  order: [],             // this tester's photo order
  answers: new Map(),    // image_path -> { species, note }
  urls: new Map(),       // image_path -> signed URL
  current: 0,
  shownAt: 0,            // when the current photo appeared
  saves: Promise.resolve(),   // saves run one at a time, in the order they were made
  unsaved: new Set(),    // image_paths whose last save failed
};
const zoom = zoomable($('frame'), $('photo'));

/* ---------- screens ---------- */
function show(id) {
  for (const s of document.querySelectorAll('main > section')) s.classList.toggle('show', s.id === id);
}

function message(title, text) {
  $('message-title').textContent = title;
  $('message-text').textContent = text;
  show('message');
}

/* ---------- theme, shared with the identifier ---------- */
if (localStorage.getItem('eggic-theme') === 'dark') document.documentElement.dataset.theme = 'dark';
$('theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  if (dark) delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = 'dark';
  localStorage.setItem('eggic-theme', dark ? 'light' : 'dark');
});

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

$('signout').addEventListener('click', () => signOut().catch(console.error));

onUserChange(user => {
  state.user = user;
  $('signout').style.display = user ? '' : 'none';
  $('progress').textContent = '';
  if (user) start().catch(err => message('Something went wrong', err.message));
  else show('login');
});

/* ---------- the test ---------- */
async function start() {
  show('loading');
  const [names, answers] = await Promise.all([listTestImages(), myAnswers()]);
  if (!names.length) {
    return message('No photos for this login', 'Ask James to check your account is set up for the test.');
  }
  state.answers = answers;
  state.order = seededShuffle(names, state.user.id);
  state.urls = await signTestImages(state.order);

  const next = state.order.findIndex(p => !state.answers.has(p));
  if (next === -1) finish();
  else go(next);
}

function go(i) {
  const { order, answers, urls } = state;
  state.current = Math.max(0, Math.min(i, order.length - 1));
  const path = order[state.current];

  zoom.reset();
  $('photo').src = urls.get(path);
  $('counter').textContent = `Photo ${state.current + 1} of ${order.length}`;
  renderProgress();
  $('note').value = answers.get(path)?.note ?? '';
  for (const b of $('choices').children) b.classList.toggle('on', b.dataset.key === answers.get(path)?.species);
  show('test');
  state.shownAt = performance.now();

  for (const p of order.slice(state.current + 1, state.current + 1 + PRELOAD)) new Image().src = urls.get(p);
}

function renderProgress() {
  const { order, answers } = state;
  $('progress').textContent = `${answers.size} of ${order.length} answered`;
  $('left').textContent = order.length - answers.size;
  $('back').disabled = neighbour(-1) === -1;
  $('next').disabled = neighbour(+1) === -1;
}

// The next photo in direction dir (+1 or -1), or -1 if there is none. With
// "Unanswered only" ticked, answered photos are passed over and the search wraps round.
function neighbour(dir) {
  const { order, answers, current } = state;
  const n = order.length;
  const only = $('only-unanswered').checked;
  for (let k = 1; k < n; k++) {
    const i = only ? ((current + dir * k) % n + n) % n : current + dir * k;
    if (i < 0 || i >= n) return -1;
    if (!only || !answers.has(order[i])) return i;
  }
  return -1;
}

function move(dir) {
  const i = neighbour(dir);
  if (i !== -1) go(i);
}

// A link that ran out mid-sitting is signed again once, then left to fail visibly.
$('photo').addEventListener('error', async () => {
  const path = state.order[state.current];
  if (!path || $('photo').dataset.retried === path) return;
  $('photo').dataset.retried = path;
  try {
    const url = await signTestImage(path);
    state.urls.set(path, url);
    $('photo').src = url;
  } catch (err) {
    console.error(err);
  }
});

// The answer counts at once and the next photo shows while it saves. Saves queue in
// order, so changing an answer can never be overwritten by the older one landing late.
function answer(species) {
  const { order, answers } = state;
  if (!order.length || performance.now() - state.shownAt < SETTLE_MS) return;
  const path = order[state.current];
  const entry = { species, note: $('note').value.trim() || null };
  answers.set(path, entry);
  state.saves = state.saves
    .then(() => saveAnswer(state.user.id, path, entry.species, entry.note))
    .then(() => saved(path), err => notSaved(path, entry, err));

  const next = neighbour(+1);
  if (next !== -1) go(next);
  else settle().catch(err => message('Something went wrong', err.message));
}

function saved(path) {
  state.unsaved.delete(path);
  if (!state.unsaved.size) $('save-error').textContent = '';
}

// A failed save leaves the photo unanswered (unless it was answered again since), so
// the end of the test brings the tester back to it. The message stays until it saves.
function notSaved(path, entry, err) {
  console.error(err);
  if (state.answers.get(path) === entry) state.answers.delete(path);
  state.unsaved.add(path);
  $('save-error').textContent = `Photo ${state.order.indexOf(path) + 1} didn't save (${err.message}). ` +
    "It's left unanswered, so you'll be brought back to it.";
  renderProgress();
}

// After the last photo: wait for every save, then finish, or turn on "Unanswered only"
// and go to the first photo that was skipped or didn't save.
async function settle() {
  show('loading');
  await state.saves;
  const gap = state.order.findIndex(p => !state.answers.has(p));
  if (gap === -1) return finish();
  $('only-unanswered').checked = true;
  go(gap);
}

function finish() {
  const { order, answers } = state;
  const unsure = [...answers.values()].filter(a => a.species === NOT_SURE).length;
  renderProgress();
  $('done-text').textContent = `You've answered all ${order.length} photos` +
    (unsure ? ` (${unsure} marked Not sure).` : '.') + ' Your answers are saved.';
  show('done');
}

/* ---------- choices and keys ---------- */
function renderChoices() {
  const options = [...SPECIES, { key: NOT_SURE, name: 'Not sure' }];
  for (const [i, sp] of options.entries()) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = sp.key === NOT_SURE ? 'choice unsure' : 'choice';
    b.dataset.key = sp.key;

    const kbd = document.createElement('kbd');
    kbd.textContent = sp.key === NOT_SURE ? '0' : String(i + 1);
    const label = document.createElement('span');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = sp.name;
    label.append(name);
    if (sp.latin) {
      const latin = document.createElement('span');
      latin.className = 'latin';
      latin.textContent = sp.latin;
      label.append(latin);
    }
    b.append(kbd, label);
    b.addEventListener('click', () => answer(sp.key));
    $('choices').append(b);
  }
}

renderChoices();
$('review').addEventListener('click', () => {
  $('only-unanswered').checked = false;
  go(0);
});
$('back').addEventListener('click', () => move(-1));
$('next').addEventListener('click', () => move(+1));
// Ticking the filter on an answered photo moves straight to the next unanswered one.
$('only-unanswered').addEventListener('change', e => {
  if (e.target.checked && state.answers.has(state.order[state.current])) move(+1);
  else renderProgress();
});

document.addEventListener('keydown', e => {
  if (!$('test').classList.contains('show') || e.metaKey || e.ctrlKey || e.altKey) return;
  // Typing a note must not answer the photo; Escape leaves the box so the keys work again.
  if (e.target.closest('textarea')) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.key >= '1' && e.key <= '9') answer(SPECIES[Number(e.key) - 1].key);
  else if (e.key === '0') answer(NOT_SURE);
  else if (e.key === 'ArrowLeft') move(-1);
  else if (e.key === 'ArrowRight') move(+1);
});
