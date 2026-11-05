// The one Supabase client in the browser. Pages never import supabase-js or
// touch `db` themselves; they call the services beside this file.
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.117.3/+esm';
import { SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY } from '../config.js';

export const db = createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

/** A supabase-js result as plain data, or a thrown Error. Services return data or throw. */
export function unwrap({ data, error }) {
  if (error) throw new Error(error.message);
  return data;
}
