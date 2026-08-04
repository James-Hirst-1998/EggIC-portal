// Vercel serverless function: browser -> here -> Modal.
//
// This hop exists so the Modal key never reaches the browser, and so the
// deployed site (unlike local development) asks Modal to keep every submission.

// A Modal cold start is ~20s; the default 10s function limit would abort it.
export const config = { maxDuration: 60 };

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' });
  }

  const { MODAL_PREDICT_URL, EGGIC_API_KEY } = process.env;
  if (!MODAL_PREDICT_URL || !EGGIC_API_KEY) {
    return res.status(500).json({ error: 'Server is not configured.' });
  }

  const { image, filename } = req.body || {};
  if (!image) {
    return res.status(400).json({ error: 'No image supplied.' });
  }

  try {
    const upstream = await fetch(MODAL_PREDICT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image,
        filename: filename || null,
        key: EGGIC_API_KEY,
        store: true,   // the deployed site keeps every submission
      }),
    });

    const text = await upstream.text();
    if (!upstream.ok) {
      return res.status(upstream.status).json({ error: text.slice(0, 500) });
    }
    res.setHeader('Content-Type', 'application/json');
    return res.status(200).send(text);
  } catch (err) {
    return res.status(502).json({ error: `Could not reach the model: ${err.message}` });
  }
}
