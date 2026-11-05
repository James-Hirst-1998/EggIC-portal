// A tester's answers. Row-level security (own_answers) limits every call here to
// the signed-in tester's own rows.
import { db, unwrap } from './supabase.js';

/** Answers so far, as a Map of image_path -> { species, note }. */
export async function myAnswers() {
  const rows = unwrap(await db.from('test_answers').select('image_path, species, note'));
  return new Map(rows.map(r => [r.image_path, { species: r.species, note: r.note }]));
}

/** Save an answer, or change one given earlier. An empty note is stored as null. */
export async function saveAnswer(userId, imagePath, species, note) {
  unwrap(await db.from('test_answers').upsert(
    { user_id: userId, image_path: imagePath, species, note: note || null },
    { onConflict: 'user_id,image_path' },
  ));
}
