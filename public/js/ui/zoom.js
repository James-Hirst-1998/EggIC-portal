// Zoom a photo inside its frame: scroll or pinch to zoom at the pointer, drag to move,
// double-click to zoom in or back out. The frame clips; the image only gets a CSS
// transform, so nothing reloads.
const MAX = 8;

export function zoomable(frame, img) {
  let s = 1, x = 0, y = 0;     // transform: translate(x, y) scale(s), origin top left
  const pointers = new Map();  // pointerId -> [x, y] in the frame
  let pinch = null;            // { d, s }: finger spread and scale when a pinch began

  function apply() {
    const w = frame.clientWidth, h = frame.clientHeight;
    x = Math.min(0, Math.max(w - w * s, x));   // never past an edge
    y = Math.min(0, Math.max(h - h * s, y));
    img.style.transform = s === 1 ? '' : `translate(${x}px, ${y}px) scale(${s})`;
    frame.classList.toggle('zoomed', s > 1);
  }

  // Zoom to scale `to`, keeping the frame point (px, py) still.
  function zoomAt(px, py, to) {
    const k = Math.min(MAX, Math.max(1, to)) / s;
    x = px - (px - x) * k;
    y = py - (py - y) * k;
    s *= k;
    apply();
  }

  function reset() {
    s = 1; x = 0; y = 0;
    apply();
  }

  const local = e => {
    const r = frame.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const two = () => [...pointers.values()];
  const spread = ([a, b] = two()) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const middle = ([a, b] = two()) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

  frame.addEventListener('wheel', e => {
    e.preventDefault();
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : 1);    // Firefox scrolls in lines
    const rate = e.ctrlKey ? 0.01 : 0.0015;                // ctrlKey: a trackpad pinch
    zoomAt(...local(e), s * Math.exp(-dy * rate));
  }, { passive: false });

  frame.addEventListener('dblclick', e => {
    if (s > 1) reset();
    else zoomAt(...local(e), 3);
  });

  frame.addEventListener('pointerdown', e => {
    frame.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, local(e));
    if (pointers.size === 2) pinch = { d: spread(), s };
  });
  frame.addEventListener('pointermove', e => {
    const was = pointers.get(e.pointerId);
    if (!was) return;
    const now = local(e);
    pointers.set(e.pointerId, now);
    if (pinch && pointers.size === 2) {
      zoomAt(...middle(), pinch.s * spread() / pinch.d);
    } else if (pointers.size === 1 && s > 1) {
      x += now[0] - was[0];
      y += now[1] - was[1];
      apply();
    }
  });
  const lift = e => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
  };
  frame.addEventListener('pointerup', lift);
  frame.addEventListener('pointercancel', lift);

  return { reset };
}
