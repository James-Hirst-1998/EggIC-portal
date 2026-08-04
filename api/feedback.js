// Vercel serverless function: records whether a call was right against the
// stored submission. The id comes from the matching /api/predict response.

export const config = { maxDuration: 30 };

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' });
  }

  const { MODAL_FEEDBACK_URL, EGGIC_API_KEY } = process.env;
  if (!MODAL_FEEDBACK_URL || !EGGIC_API_KEY) {
    return res.status(500).json({ error: 'Server is not configured.' });
  }

  const { id, correct, user_species } = req.body || {};
  if (!id) {
    // Nothing was stored for this photo, so there is nothing to attach to.
    return res.status(200).json({ stored: false, reason: 'no submission id' });
  }

  try {
    const upstream = await fetch(MODAL_FEEDBACK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id,
        correct: correct ?? null,
        user_species: user_species ?? null,
        key: EGGIC_API_KEY,
      }),
    });

    const text = await upstream.text();
    if (!upstream.ok) {
      return res.status(upstream.status).json({ stored: false, error: text.slice(0, 500) });
    }
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).send(text);
  } catch (err) {
    return res.status(502).json({ stored: false, error: err.message });
  }
}
