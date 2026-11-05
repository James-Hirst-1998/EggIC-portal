// A tester's answers. Row-level security (own_answers) limits every call here to
// the signed-in tester's own rows.
import { db, unwrap } from './supabase.js';

/** Answers so far, as a Map of image_path -> species. */
export async function myAnswers() {
  const rows = unwrap(await db.from('test_answers').select('image_path, species'));
  return new Map(rows.map(r => [r.image_path, r.species]));
}

/** Save an answer, or change one given earlier. */
export async function saveAnswer(userId, imagePath, species) {
  unwrap(await db.from('test_answers').upsert(
    { user_id: userId, image_path: imagePath, species },
    { onConflict: 'user_id,image_path' },
  ));
}
