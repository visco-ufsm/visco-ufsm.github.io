import { useEffect, useRef } from "react";

/* The hero visual, deliberately not tied to a single research line.
 *
 * A surface sampled on a regular lattice and drawn as square samples: what a
 * depth sensor, a reconstruction and a coded frame all reduce to. Rotating, the
 * lattice reads as structure rather than noise, and the squares read as pixels.
 *
 * The pointer raises the samples under it into a mound that follows the cursor
 * and decays with distance, so the surface answers to the reader.
 *
 * Drawing thousands of samples every frame only stays cheap under three rules:
 * no allocation inside the loop, no trigonometry per sample, and one fillStyle
 * per colour rather than one per sample. `draw` keeps all three. */

const SPAN = 1.5; // half-width of the sampled footprint, in world units
const FADE_FROM = 0.72; // samples start dimming here, reaching zero at SPAN
const N = 67; // lattice is N x N across 2 * SPAN
const TILT = 0.52; // radians
const FOCAL = 3.6;

const MOUND_RADIUS = 118; // px on screen
const MOUND_HEIGHT = 0.09; // world units — a swell, not a spike
const ACCENT_RATE = 0.012;
const ACCENTS = ["#5b4bd6", "#0e8c7a", "#c4568a"];

/* Samples are bucketed by colour and brightness so the draw loop sets fillStyle
   once per bucket. 40 steps is past the point where banding shows. */
const STEPS = 40;
const LEVELS = STEPS + 1;
const COLOURS = 1 + ACCENTS.length; // ink, then each accent
const BUCKETS = COLOURS * LEVELS;

const FILL: string[] = [];
for (let i = 0; i < LEVELS; i++) FILL.push(`rgba(11, 14, 20, ${(i / STEPS) * 0.62})`);
for (const c of ACCENTS)
  for (let i = 0; i < LEVELS; i++)
    FILL.push(
      c +
        Math.round(Math.min(1, (i / STEPS) * 1.9) * 255)
          .toString(16)
          .padStart(2, "0"),
    );

export default function PointCloud({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    /* ── Lattice, resolved once ──────────────────────────────────────────────
       The height field is 0.15·sin(2.6x + t) + 0.10·cos(2.1z − 0.75t)
       + 0.055·sin(3.9(x+z) + 0.5t). Expanding each term with the angle
       addition rule separates a part that depends only on the sample from a
       part that depends only on time, so all per-sample trigonometry happens
       here instead of on every frame. */
    const cols: number[][] = [[], [], [], [], [], [], [], [], [], []];
    const [xs, zs, fades, accs, h1s, h1c, h2s, h2c, h3s, h3c] = cols;

    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 2 * SPAN - SPAN;
      for (let j = 0; j < N; j++) {
        const z = (j / (N - 1)) * 2 * SPAN - SPAN;
        const d = Math.hypot(x, z);
        if (d > SPAN) continue;
        // Dissolve toward the rim instead of ending on a visible edge.
        const f = Math.min(1, Math.max(0, (SPAN - d) / (SPAN - FADE_FROM)));
        const a1 = 2.6 * x;
        const a2 = 2.1 * z;
        const a3 = 3.9 * (x + z);
        xs.push(x);
        zs.push(z);
        fades.push(f * f);
        accs.push(Math.random() < ACCENT_RATE ? (Math.random() * 3) | 0 : -1);
        h1s.push(0.15 * Math.sin(a1));
        h1c.push(0.15 * Math.cos(a1));
        h2s.push(0.1 * Math.sin(a2));
        h2c.push(0.1 * Math.cos(a2));
        h3s.push(0.055 * Math.sin(a3));
        h3c.push(0.055 * Math.cos(a3));
      }
    }

    const count = xs.length;
    const X = Float32Array.from(xs);
    const Z = Float32Array.from(zs);
    const FADE = Float32Array.from(fades);
    const ACC = Int8Array.from(accs);
    const H1S = Float32Array.from(h1s);
    const H1C = Float32Array.from(h1c);
    const H2S = Float32Array.from(h2s);
    const H2C = Float32Array.from(h2c);
    const H3S = Float32Array.from(h3s);
    const H3C = Float32Array.from(h3c);

    // Scratch buffers, reused every frame so the draw loop never allocates.
    const sxA = new Float32Array(count);
    const syA = new Float32Array(count);
    const szA = new Float32Array(count);
    const bkA = new Int16Array(count);
    const outX = new Float32Array(count);
    const outY = new Float32Array(count);
    const outS = new Float32Array(count);
    const counts = new Int32Array(BUCKETS);
    const starts = new Int32Array(BUCKETS);

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let raf = 0;
    let visible = true;
    let t = 0;
    let last = performance.now();

    // Pointer, smoothed so the mound trails the cursor slightly.
    let targetX = -9999;
    let targetY = -9999;
    let cursorX = -9999;
    let cursorY = -9999;
    let strength = 0;
    let targetStrength = 0;

    const size = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      w = r.width;
      h = r.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    /* The canvas sits inside a pointer-events-none layer, so the cursor is
       tracked on the window and converted into canvas coordinates. */
    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      targetX = e.clientX - r.left;
      targetY = e.clientY - r.top;
      if (cursorX < -9000) {
        cursorX = targetX;
        cursorY = targetY;
      }
      targetStrength = 1;
    };
    const onLeave = () => {
      targetStrength = 0;
    };

    const draw = () => {
      if (w === 0 || h === 0) return;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w * 0.38, h * 0.78);
      const spin = t * 0.16;
      const cosA = Math.cos(spin);
      const sinA = Math.sin(spin);
      const cosT = Math.cos(TILT);
      const sinT = Math.sin(TILT);
      // The only trigonometry per frame: the time half of the height field.
      const ct = Math.cos(t);
      const st = Math.sin(t);
      const c75 = Math.cos(0.75 * t);
      const s75 = Math.sin(0.75 * t);
      const c5 = Math.cos(0.5 * t);
      const s5 = Math.sin(0.5 * t);

      const r2 = MOUND_RADIUS * MOUND_RADIUS;
      const lifting = strength > 0.003 && cursorX > -9000;
      /* A lifted sample is drawn this many pixels higher than where it sits on
         the surface, scaled by its own perspective factor. Measuring the mound
         from that far *below* the cursor puts its peak under the cursor rather
         than above it. */
      const liftPx = MOUND_HEIGHT * strength * cosT * scale;

      counts.fill(0);
      let n = 0;

      for (let i = 0; i < count; i++) {
        const x = X[i];
        const z = Z[i];
        const rx = x * cosA + z * sinA;
        const rz = -x * sinA + z * cosA;
        let y =
          H1S[i] * ct +
          H1C[i] * st +
          H2C[i] * c75 +
          H2S[i] * s75 +
          H3S[i] * c5 +
          H3C[i] * s5;

        // First projection locates the sample on screen...
        let ry = y * cosT - rz * sinT;
        let rz2 = y * sinT + rz * cosT;
        let k = FOCAL / (FOCAL + rz2);
        let sx = cx + rx * k * scale;
        let sy = cy + ry * k * scale;

        // ...so the mound can be measured in the space the reader sees, then
        // applied to the height and the sample projected again.
        if (lifting) {
          const dx = sx - cursorX;
          const dy = sy - (cursorY + liftPx * k);
          const q = (dx * dx + dy * dy) / r2;
          if (q < 9) {
            y -= MOUND_HEIGHT * Math.exp(-q) * strength;
            ry = y * cosT - rz * sinT;
            rz2 = y * sinT + rz * cosT;
            k = FOCAL / (FOCAL + rz2);
            sx = cx + rx * k * scale;
            sy = cy + ry * k * scale;
          }
        }

        if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) continue;

        // Samples the reader sees as higher are brighter, so the surface reads
        // as shaded rather than flat.
        const lift = (0.3 - y) / 0.6;
        let a = ((k - 0.72) * 1.9 + lift * 0.12) * FADE[i];
        if (a <= 0.002) continue;
        if (a > 1) a = 1;

        const size2 = k * 1.7;
        const bucket = (ACC[i] < 0 ? 0 : (ACC[i] + 1) * LEVELS) + ((a * STEPS) | 0);
        sxA[n] = sx;
        syA[n] = sy;
        szA[n] = size2 < 0.7 ? 0.7 : size2;
        bkA[n] = bucket;
        counts[bucket]++;
        n++;
      }

      /* Counting sort into bucket order: linear and comparator-free. It also
         puts dim samples before bright ones, which is the far-to-near order the
         surface needs, so it replaces the depth sort outright. */
      let acc = 0;
      for (let b = 0; b < BUCKETS; b++) {
        starts[b] = acc;
        acc += counts[b];
      }
      for (let i = 0; i < n; i++) {
        const at = starts[bkA[i]]++;
        outX[at] = sxA[i];
        outY[at] = syA[i];
        outS[at] = szA[i];
      }

      // One fillStyle per bucket instead of one per sample.
      let from = 0;
      for (let b = 0; b < BUCKETS; b++) {
        const c = counts[b];
        if (c === 0) continue;
        ctx.fillStyle = FILL[b];
        const to = from + c;
        for (let i = from; i < to; i++)
          ctx.fillRect(outX[i], outY[i], outS[i], outS[i]);
        from = to;
      }
    };

    const frame = (now: number) => {
      const dt = Math.min(now - last, 64);
      last = now;
      if (visible) {
        t += dt * 0.00022;
        cursorX += (targetX - cursorX) * 0.2;
        cursorY += (targetY - cursorY) * 0.2;
        strength += (targetStrength - strength) * 0.08;
        draw();
      }
      raf = requestAnimationFrame(frame);
    };

    size();
    draw();

    const ro = new ResizeObserver(() => {
      size();
      draw();
    });
    ro.observe(canvas);

    const io = new IntersectionObserver(
      ([e]) => {
        visible = e.isIntersecting;
      },
      { threshold: 0 },
    );
    io.observe(canvas);

    if (!still) {
      window.addEventListener("pointermove", onMove, { passive: true });
      document.addEventListener("pointerleave", onLeave);
      raf = requestAnimationFrame(frame);
    }

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} aria-hidden="true" className={className} />;
}
