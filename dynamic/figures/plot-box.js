/* ══════════════════════════════════════════════════════════════
 * SHARED · 2D PLOT BOX   (used by the Θ-space figures)
 *
 * Draws a square parameter-space box like the paper's Fig. 2 panels:
 * a title above ("Θ-space"), a frame, and axis labels sym₁ (horizontal)
 * and sym₂ (vertical), with no ticks. Returns the data → SVG maps, a
 * clipped layer for contours and a clipped layer for markers.
 * Styles: style.css Section 15.6 (.tp-*).
 * ══════════════════════════════════════════════════════════════ */
window.PlotBox = function (svg, { id, title, sym, view }) {
  const { el } = window.TwoStateMDP;
  const P = { x0: 34, y0: 30, size: 256 };                                     // box (SVG units)

  const X = t => P.x0 + (t - view.x[0]) / (view.x[1] - view.x[0]) * P.size;
  const Y = t => P.y0 + (view.y[1] - t) / (view.y[1] - view.y[0]) * P.size;
  const invX = x => view.x[0] + (x - P.x0) / P.size * (view.x[1] - view.x[0]);
  const invY = y => view.y[1] - (y - P.y0) / P.size * (view.y[1] - view.y[0]);

  const clip = el('clipPath', { id: id + '-clip' }, el('defs', {}, svg));
  el('rect', { x: P.x0, y: P.y0, width: P.size, height: P.size }, clip);

  if (title) el('text', { x: P.x0 + P.size / 2, y: P.y0 - 10, class: 'tp-title', 'text-anchor': 'middle' }, svg)
    .textContent = title;                                                       // '' → no title (e.g. a menu instead)
  el('rect', { x: P.x0, y: P.y0, width: P.size, height: P.size, class: 'tp-frame' }, svg);
  const under = el('g', { 'clip-path': 'url(#' + id + '-clip)' }, svg);          // contours etc.
  const marks = el('g', { 'clip-path': 'url(#' + id + '-clip)', class: 'tp-marks' }, svg);

  const axis = (x, y, k, rotate) => el('text', Object.assign({ x, y, class: 'tp-axis', 'text-anchor': 'middle' },
    rotate ? { transform: 'rotate(-90 ' + x + ' ' + y + ')' } : {}), svg)
    .innerHTML = '<tspan font-style="italic">' + sym + '</tspan><tspan dy="4" class="tp-sub">' + k + '</tspan>';
  axis(P.x0 + P.size / 2, P.y0 + P.size + 25, 1, false);
  axis(14, P.y0 + P.size / 2, 2, true);

  // Pointer position in data coordinates, or null outside the box
  const toData = ev => {
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const inside = p.x >= P.x0 && p.x <= P.x0 + P.size && p.y >= P.y0 && p.y <= P.y0 + P.size;
    return inside ? [invX(p.x), invY(p.y)] : null;
  };

  // A labelled point: shape 'circle' (Θ, as the θ path in geometries.py) or 'square' (Φ, as in
  // the paper's Fig. 1). `label` is from PlotBox.label(); `side` puts it 'left', 'right' or
  // 'below' the point (use it to keep labels of nearby points apart).
  // Returns move([x, y]) and show(bool).
  function point(xy, { shape = 'circle', label = '', side = 'right', cls = '' }) {
    const g = el('g', { class: 'tp-pt ' + cls }, marks);
    const m = shape === 'square'
      ? el('rect', { width: 8, height: 8, class: 'tp-theta' }, g)
      : el('circle', { r: 4.5, class: 'tp-theta' }, g);
    const t = el('text', { class: 'tp-label', 'text-anchor': { left: 'end', right: 'start', below: 'middle' }[side] }, g);
    t.innerHTML = label;
    // A hat is drawn as a small caret over the first glyph, measured from the rendered text,
    // because combining accents (U+0302) sit badly on fallback-font glyphs such as ϕ.
    const hat = t.querySelector('.tp-hatbase') ? el('path', { class: 'tp-hat' }, g) : null;
    const placeHat = () => {
      if (!hat) return;
      try {
        const b = t.getExtentOfChar(0), cx = b.x + b.width * 0.58, top = b.y + b.height * 0.1;
        hat.setAttribute('d', 'M' + (cx - 3.4) + ' ' + (top + 3) + 'L' + cx + ' ' + top + 'L' + (cx + 3.4) + ' ' + (top + 3));
      } catch (e) { /* not rendered yet (hidden) */ }
    };
    const move = ([x, y]) => {
      const px = X(x), py = Y(y);
      if (shape === 'square') { m.setAttribute('x', px - 4); m.setAttribute('y', py - 4); }
      else { m.setAttribute('cx', px); m.setAttribute('cy', py); }
      t.setAttribute('x', px + { left: -7, right: 7, below: 0 }[side]);
      t.setAttribute('y', py + (side === 'below' ? 22 : -9));
      placeHat();
    };
    move(xy);
    return { move, show: on => { g.classList.toggle('is-hidden', !on); if (on) placeHat(); } };
  }

  return { X, Y, toData, under, marks, point };
};

// Label markup for a symbol with optional hat, subscript, superscript and a superscript on
// the subscript, e.g. label('ϕ', { sub: 'ρ', subsup: '(i)' }) → ϕ_{ρ^{(i)}}.
window.PlotBox.label = (base, { hat = false, sub = '', subsup = '', sup = '' } = {}) =>
  '<tspan font-style="italic"' + (hat ? ' class="tp-hatbase"' : '') + '>' + base + '</tspan>' +
  (sup ? '<tspan dy="-6" class="tp-sup">' + sup + '</tspan>' : '') +
  (sub ? '<tspan dy="5" class="tp-sup" font-style="italic">' + sub + '</tspan>' : '') +
  (subsup ? '<tspan dy="-6" class="tp-supsub">' + subsup + '</tspan>' : '');
