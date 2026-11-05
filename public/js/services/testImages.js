// The staff-test photos. Testers can list and view them (storage policy
// read_test_images); the answer key in the test_images table is never readable here.
import { db, unwrap } from './supabase.js';
import { TEST_BUCKET } from '../config.js';

const LINK_TTL_S = 6 * 60 * 60;   // long enough for a whole sitting

/** Every test photo name, sorted. Empty for an account that isn't a tester. */
export async function listTestImages() {
  const files = unwrap(await db.storage.from(TEST_BUCKET).list('', { limit: 1000 }));
  return files.map(f => f.name).filter(name => name.endsWith('.jpg')).sort();
}

/** Signed links for many photos in one call, as a Map of name -> URL. */
export async function signTestImages(paths) {
  const signed = unwrap(await db.storage.from(TEST_BUCKET).createSignedUrls(paths, LINK_TTL_S));
  return new Map(signed.map(s => [s.path, s.signedUrl]));
}

/** A fresh signed link for one photo, for when an old one has run out. */
export async function signTestImage(path) {
  return unwrap(await db.storage.from(TEST_BUCKET).createSignedUrl(path, LINK_TTL_S)).signedUrl;
}
