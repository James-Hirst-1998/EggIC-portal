// Staff test page: one photo at a time, nine species plus Not sure.
// This file owns the page; every Supabase call goes through ./services.
import { signIn, signOut, onUserChange } from './services/auth.js';
import { listTestImages, signTestImages, signTestImage } from './services/testImages.js';
import { myAnswers, saveAnswer } from './services/testAnswers.js';
import { SPECIES, NOT_SURE } from './species.js';
import { seededShuffle } from './lib/seededShuffle.js';

const $ = id => document.getElementById(id);

const state = {
  user: null,
  order: [],             // this tester's photo order
  answers: new Map(),    // image_path -> species
  urls: new Map(),       // image_path -> signed URL
  current: 0,
  saving: false,
};

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

  $('photo').src = urls.get(path);
  $('full').href = urls.get(path);
  $('back').disabled = state.current === 0;
  $('counter').textContent = `Photo ${state.current + 1} of ${order.length}`;
  $('progress').textContent = `${answers.size} of ${order.length} answered`;
  $('save-error').textContent = '';
  for (const b of $('choices').children) b.classList.toggle('on', b.dataset.key === answers.get(path));
  show('test');

  const after = order[state.current + 1];
  if (after) new Image().src = urls.get(after);   // so the next photo is already here
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

async function answer(species) {
  const { order, answers } = state;
  if (state.saving || !order.length) return;
  const path = order[state.current];

  state.saving = true;
  setChoicesDisabled(true);
  try {
    await saveAnswer(state.user.id, path, species);
  } catch (err) {
    $('save-error').textContent = `Not saved: ${err.message}. Try again.`;
    return;
  } finally {
    state.saving = false;
    setChoicesDisabled(false);
  }

  answers.set(path, species);
  if (state.current < order.length - 1) return go(state.current + 1);
  const gap = order.findIndex(p => !answers.has(p));
  if (gap === -1) finish();
  else go(gap);
}

function finish() {
  const { order, answers } = state;
  const unsure = [...answers.values()].filter(s => s === NOT_SURE).length;
  $('progress').textContent = `${answers.size} of ${order.length} answered`;
  $('done-text').textContent = `You've answered all ${order.length} photos` +
    (unsure ? ` (${unsure} marked Not sure).` : '.') + ' Your answers are saved.';
  show('done');
}

/* ---------- choices and keys ---------- */
function setChoicesDisabled(disabled) {
  for (const b of $('choices').children) b.disabled = disabled;
}

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
$('review').addEventListener('click', () => go(0));
$('back').addEventListener('click', () => go(state.current - 1));

document.addEventListener('keydown', e => {
  if (!$('test').classList.contains('show') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key >= '1' && e.key <= '9') answer(SPECIES[Number(e.key) - 1].key);
  else if (e.key === '0') answer(NOT_SURE);
  else if (e.key === 'ArrowLeft') go(state.current - 1);
  else if (e.key === 'ArrowRight' && state.answers.has(state.order[state.current])) go(state.current + 1);
});
