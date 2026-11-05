// Logins. Accounts are made by scripts/add_tester.py; there is no sign-up here.
import { db, unwrap } from './supabase.js';

export async function signIn(email, password) {
  unwrap(await db.auth.signInWithPassword({ email, password }));
}

export async function signOut() {
  unwrap(await db.auth.signOut());
}

/**
 * Calls back with the signed-in user, or null: once on load, then on each change.
 * Token refreshes for the same user are not changes. The callback is deferred
 * because supabase-js deadlocks if its own calls run inside the listener.
 */
export function onUserChange(callback) {
  let lastId;
  db.auth.onAuthStateChange((_event, session) => {
    const user = session?.user ?? null;
    if ((user?.id ?? null) === lastId) return;
    lastId = user?.id ?? null;
    setTimeout(() => callback(user));
  });
}
