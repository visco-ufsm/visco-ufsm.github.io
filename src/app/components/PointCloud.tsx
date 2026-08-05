import { useEffect, useRef } from "react";

/* An alternative hero visual, deliberately not tied to one research line.
 *
 * A surface sampled on a regular lattice and drawn as square samples: what a
 * depth sensor, a reconstruction or a coded frame all reduce to. Rotating, the
 * lattice reads as structure rather than noise, and the squares read as pixels.
 * Generic enough for compression, vision, processing and mining alike.
 *
 * The pointer steers it the way a viewport is steered inside a 360° scene:
 * horizontal position is yaw, vertical position is pitch, and the idle spin
 * eases off while you hold the controls. The camera carries mass, so it trails
 * the cursor instead of tracking it. */

const N = 56; // lattice is N x N
const TILT = 0.52; // radians
const FOCAL = 3.2;
const ACCENTS = ["#5b4bd6", "#0e8c7a", "#c4568a"];

const YAW = 0.62; // rad the pointer can add to the rotation
/* Pitch travel is asymmetric: raising the camera opens the lattice up, but
   lowering it past ~0.42 rad flattens the surface to edge-on, where the
   height field compresses into a dark band and stops reading as a surface. */
const PITCH_UP = 0.26;
const PITCH_DOWN = 0.1;
const TAU = 190; // ms; how long the camera takes to catch up
const DRIFT = 0.55; // fraction of the idle spin surrendered to the pointer

/** Smooth rolling height field. Deterministic, so it never looks random. */
function height(x: number, z: number, t: number) {
  return (
    0.15 * Math.sin(2.6 * x + t) +
    0.1 * Math.cos(2.1 * z - t * 0.75) +
    0.055 * Math.sin(3.9 * (x + z) + t * 0.5)
  );
}

type Sample = { sx: number; sy: number; s: number; a: number; c: string | null };

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  t: number,
  spin: number,
  tilt: number,
  accents: (string | null)[],
) {
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2;
  const scale = Math.min(w, h * 1.8) * 0.44;
  const cosA = Math.cos(spin);
  const sinA = Math.sin(spin);
  const cosT = Math.cos(tilt);
  const sinT = Math.sin(tilt);

  const out: Sample[] = [];

  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    for (let j = 0; j < N; j++) {
      const z = (j / (N - 1)) * 2 - 1;
      // Round the footprint so the lattice reads as a disc, not a square slab.
      const d = Math.hypot(x, z);
      if (d > 1) continue;

      const y = height(x, z, t);
      const rx = x * cosA + z * sinA;
      const rz = -x * sinA + z * cosA;
      const ry = y * cosT - rz * sinT;
      const rz2 = y * sinT + rz * cosT;

      const k = FOCAL / (FOCAL + rz2);
      const sx = cx + rx * k * scale;
      const sy = cy + ry * k * scale;
      if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) continue;

      // Crests read brighter, so the surface has shading instead of flat dots.
      const lift = (y + 0.3) / 0.6;
      const edge = 1 - d * 0.55;
      out.push({
        sx,
        sy,
        s: Math.max(0.7, k * 1.7),
        a: Math.min(0.62, Math.max(0.04, (k - 0.72) * 1.9 * edge + lift * 0.12)),
        c: accents[i * N + j],
      });
    }
  }

  out.sort((p, q) => p.s - q.s); // far samples first

  for (const p of out) {
    ctx.fillStyle = p.c
      ? `${p.c}${Math.round(Math.min(1, p.a * 1.9) * 255)
          .toString(16)
          .padStart(2, "0")}`
      : `rgba(11, 14, 20, ${p.a})`;
    ctx.fillRect(p.sx, p.sy, p.s, p.s);
  }
}

export default function PointCloud({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Fixed accent positions, so they do not flicker between frames.
    const accents = Array.from({ length: N * N }, () =>
      Math.random() < 0.012 ? ACCENTS[(Math.random() * 3) | 0] : null,
    );

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let raf = 0;
    let visible = true;
    let t = 0; // height-field clock, never steered
    let spin = 0; // accumulated idle rotation
    let last = performance.now();

    /* Where the pointer wants the camera, and where the camera actually is. */
    let wantYaw = 0;
    let wantPitch = 0;
    let wantGrip = 0;
    let yaw = 0;
    let pitch = 0;
    let grip = 0;

    const size = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      w = r.width;
      h = r.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const paint = () => {
      if (w > 0 && h > 0)
        draw(ctx, w, h, t, spin * 0.16 + yaw, TILT + pitch, accents);
    };

    const frame = (now: number) => {
      const dt = Math.min(now - last, 64);
      last = now;
      if (visible) {
        t += dt * 0.00022;
        // The idle spin surrenders to the pointer, then takes over again.
        spin += dt * 0.00022 * (1 - DRIFT * grip);

        const k = 1 - Math.exp(-dt / TAU);
        yaw += (wantYaw - yaw) * k;
        pitch += (wantPitch - pitch) * k;
        grip += (wantGrip - grip) * k;

        paint();
      }
      raf = requestAnimationFrame(frame);
    };

    size();
    paint();

    const ro = new ResizeObserver(() => {
      size();
      paint();
    });
    ro.observe(canvas);

    const io = new IntersectionObserver(
      ([e]) => {
        visible = e.isIntersecting;
      },
      { threshold: 0 },
    );
    io.observe(canvas);

    /* The canvas sits under a pointer-events-none mask, so the pointer is
       measured against the drawn disc rather than tracked on the element.
       Influence is strongest over the lattice and fades out within REACH. */
    const REACH = 200; // px of pull beyond the disc's edge

    const onMove = (e: MouseEvent) => {
      if (!visible) return; // no layout reads once the hero is scrolled past
      const r = canvas.getBoundingClientRect();
      if (!r.width || !r.height) return;

      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const radius = Math.min(r.width, r.height * 1.8) * 0.44;
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;

      const reach = Math.max(0, Math.hypot(dx, dy) - radius);
      const near = Math.max(0, 1 - reach / REACH);
      wantGrip = near * near * (3 - 2 * near); // smoothstep, soft on approach

      const ny = Math.max(-1, Math.min(1, dy / radius));
      wantYaw = Math.max(-1, Math.min(1, dx / radius)) * YAW * wantGrip;
      wantPitch = -ny * (ny < 0 ? PITCH_UP : PITCH_DOWN) * wantGrip;
    };

    const onLeave = () => {
      wantYaw = 0;
      wantPitch = 0;
      wantGrip = 0;
    };

    if (!still) {
      raf = requestAnimationFrame(frame);
      window.addEventListener("mousemove", onMove, { passive: true });
      document.addEventListener("mouseleave", onLeave);
    }

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} aria-hidden="true" className={className} />;
}
