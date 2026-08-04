import { useEffect, useRef } from "react";
import { PLACE } from "../data";

/* The signature element: the graticule of a tilted, slowly turning sphere drawn
 * in equirectangular projection, the exact transform VisCo's work is about.
 * Parallels bow, meridians crowd toward the poles, and the campus sits on it as
 * a real datum. */

const RULE = "rgba(11, 14, 20, 0.17)";
const RULE_SOFT = "rgba(11, 14, 20, 0.09)";
const IRIS = "#5b4bd6";
const TILT = 0.42; // radians; enough curve to read as a sphere, not a grid
const D2R = Math.PI / 180;

function spectrum(ctx: CanvasRenderingContext2D, w: number) {
  const g = ctx.createLinearGradient(0, 0, w, 0);
  g.addColorStop(0, "#5b4bd6");
  g.addColorStop(0.34, "#7e86e6");
  g.addColorStop(0.68, "#37b39d");
  g.addColorStop(1, "#e7a6c4");
  return g;
}

/** Project (lon, lat) on a sphere yawed by `yaw` and tilted by TILT.
 *  `h` is the height of the 2:1 band, `oy` its top edge. The projection keeps
 *  its true aspect ratio no matter how tall the canvas is. */
function project(
  lon: number,
  lat: number,
  yaw: number,
  w: number,
  h: number,
  oy: number,
) {
  const l = (lon + yaw) * D2R;
  const p = lat * D2R;
  const cp = Math.cos(p);
  const x0 = cp * Math.cos(l);
  const y0 = cp * Math.sin(l);
  const z0 = Math.sin(p);
  const y1 = y0 * Math.cos(TILT) - z0 * Math.sin(TILT);
  const z1 = y0 * Math.sin(TILT) + z0 * Math.cos(TILT);
  const lam = Math.atan2(y1, x0);
  const phi = Math.asin(Math.max(-1, Math.min(1, z1)));
  return {
    x: (lam / Math.PI) * (w / 2) + w / 2,
    y: oy + h / 2 - (phi / (Math.PI / 2)) * (h / 2),
  };
}

/** Polyline that breaks at the ±180° seam instead of streaking across. */
function stroke(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  w: number,
) {
  ctx.beginPath();
  let down = false;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (i > 0 && Math.abs(p.x - pts[i - 1].x) > w / 2) down = false;
    if (!down) {
      ctx.moveTo(p.x, p.y);
      down = true;
    } else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

function drawSphere(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  yaw: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 1;

  // Hold the 2:1 ratio of an equirectangular frame, centred in the canvas.
  const bh = Math.min(h, w / 2);
  const oy = (h - bh) / 2;

  // Parallels
  for (let lat = -75; lat <= 75; lat += 15) {
    const pts = [];
    for (let lon = -180; lon <= 180; lon += 2)
      pts.push(project(lon, lat, yaw, w, bh, oy));
    ctx.strokeStyle = lat === 0 ? RULE : RULE_SOFT;
    stroke(ctx, pts, w);
  }

  // Meridians
  for (let lon = -180; lon < 180; lon += 15) {
    const pts = [];
    for (let lat = -90; lat <= 90; lat += 2)
      pts.push(project(lon, lat, yaw, w, bh, oy));
    ctx.strokeStyle = RULE_SOFT;
    stroke(ctx, pts, w);
  }

  // The one lit meridian: the one the group stands on.
  const lit = [];
  for (let lat = -90; lat <= 90; lat += 1)
    lit.push(project(PLACE.lon, lat, yaw, w, bh, oy));
  ctx.strokeStyle = spectrum(ctx, w);
  ctx.lineWidth = 1.6;
  stroke(ctx, lit, w);

  // The campus.
  const p = project(PLACE.lon, PLACE.lat, yaw, w, bh, oy);
  ctx.beginPath();
  ctx.arc(p.x, p.y, 20, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(91, 75, 214, 0.08)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 10.5, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(91, 75, 214, 0.45)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 3.6, 0, Math.PI * 2);
  ctx.fillStyle = IRIS;
  ctx.fill();
}

export default function Graticule({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let raf = 0;
    let visible = true;
    let yaw = 0;
    let last = performance.now();

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
      if (w === 0 || h === 0) return;
      drawSphere(ctx, w, h, yaw);
    };

    const frame = (now: number) => {
      const dt = Math.min(now - last, 64);
      last = now;
      if (visible) {
        yaw = (yaw + dt * 0.0022) % 360; // ~one turn every 45 s
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

    if (!still) raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, []);

  return <canvas ref={ref} aria-hidden="true" className={className} />;
}
