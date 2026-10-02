/* ══════════════════════════════════════════════════════════════
 * SHARED · POLICY PAD   (used by occupancy.js, occupancy-match.js, concavity.js)
 *
 * A small square that sets a two-state, two-action policy with one draggable point:
 *   x = π(a=1|s=0),  y = π(a=1|s=1),  both in [0, 1].
 * The title reads π(a|s) (or a given title); the axes are labelled s = 0 and s = 1, and
 * each axis end shows the current probability of the action at that end.
 *
 *   const pad = PolicyPad(container, { value: [0.5, 0.5], onChange: p => …, title: 'π(a|s)', titleIn: el });
 *   pad.get()  → [π(a=1|s=0), π(a=1|s=1)]
 *
 * Styles: style.css (.pp-*; the title uses .oc-title, the site's DM Mono figure title).
 * ══════════════════════════════════════════════════════════════ */
window.PolicyPad = function (container, { value = [0.5, 0.5], onChange = () => {}, title = 'π(a|s)', titleIn = null } = {}) {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  };
  // The square, in a 200-wide viewBox whose vertical centre is the square's centre (so a pad centred in
  // its cell centres the square itself); its horizontal centre is at 121/200, see .pp-title in style.css
  const B = { x0: 62, y0: 12, s: 118 };
  const X = p => B.x0 + p * B.s, Y = p => B.y0 + (1 - p) * B.s;
  const clamp = v => Math.min(1, Math.max(0, v));
  const b = t => '<tspan font-weight="700">' + t + '</tspan>';    // bold upright a, s

  const root = document.createElement('div');
  root.className = 'pp';
  const head = document.createElement('div');                   // title, centred over the square
  head.className = 'oc-title pp-title';
  head.innerHTML = title;
  (titleIn || root).appendChild(head);                           // titleIn: put it in another element (e.g. a titles row)
  container.appendChild(root);
  // Title in another element (pad centred on rows): symmetric viewBox around the square.
  // Title inside the pad: no headroom above the square, so the title sits right on it.
  const svg = el('svg', { viewBox: titleIn ? '0 -33 200 208' : '0 2 200 176', class: 'pp-svg', role: 'slider',
                          'aria-label': 'Policy: horizontal is the probability of action 1 in state 0, vertical in state 1' }, root);

  el('rect', { x: B.x0, y: B.y0, width: B.s, height: B.s, class: 'pp-frame' }, svg);
  // Axis labels
  el('text', { x: B.x0 + B.s / 2, y: B.y0 + B.s + 40, class: 'pp-axis', 'text-anchor': 'middle' }, svg).innerHTML = b('s') + ' = 0';
  el('text', { x: 12, y: B.y0 + B.s / 2, class: 'pp-axis', 'text-anchor': 'middle',
               transform: 'rotate(-90 12 ' + (B.y0 + B.s / 2) + ')' }, svg).innerHTML = b('s') + ' = 1';
  // Axis ends: the probability of the action at that end (x: under the corners; y: left of the corners)
  const end = (x, y, anchor) => el('text', { x, y: y + 6, class: 'pp-prob', 'text-anchor': anchor }, svg);
  const xp0 = end(B.x0, B.y0 + B.s + 14, 'start'), xp1 = end(B.x0 + B.s, B.y0 + B.s + 14, 'end');
  const yp1 = end(B.x0 - 6, B.y0 + 8, 'end'),          yp0 = end(B.x0 - 6, B.y0 + B.s - 12, 'end');

  const dot = el('circle', { r: 6.5, class: 'pp-dot' }, svg);
  const p = [clamp(value[0]), clamp(value[1])];

  function draw() {
    dot.setAttribute('cx', X(p[0])); dot.setAttribute('cy', Y(p[1]));
    xp0.textContent = (1 - p[0]).toFixed(2); xp1.textContent = p[0].toFixed(2);
    yp0.textContent = (1 - p[1]).toFixed(2); yp1.textContent = p[1].toFixed(2);
  }

  // Drag anywhere in the square (a click jumps the point there); keyboard: arrows
  let dragging = false;
  const at = ev => {
    const q = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    p[0] = clamp((q.x - B.x0) / B.s); p[1] = clamp(1 - (q.y - B.y0) / B.s);
    draw(); onChange(p.slice());
  };
  svg.addEventListener('pointerdown', ev => { ev.preventDefault(); dragging = true; svg.setPointerCapture(ev.pointerId); at(ev); });
  svg.addEventListener('pointermove', ev => { if (dragging) at(ev); });
  ['pointerup', 'pointercancel'].forEach(t => svg.addEventListener(t, () => { dragging = false; }));
  svg.setAttribute('tabindex', 0);
  svg.addEventListener('keydown', ev => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowDown: [0, -1], ArrowUp: [0, 1] }[ev.key];
    if (!d) return;
    ev.preventDefault();
    p[0] = clamp(+(p[0] + 0.01 * d[0]).toFixed(2)); p[1] = clamp(+(p[1] + 0.01 * d[1]).toFixed(2));
    draw(); onChange(p.slice());
  });

  draw();
  return { get: () => p.slice(), el: root };
};
