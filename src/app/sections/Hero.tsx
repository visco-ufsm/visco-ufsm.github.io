import { useEffect, useRef } from "react";
import { ArrowDown } from "lucide-react";
import PointCloud from "../components/PointCloud";
import { FACULTY, GROUP, LINES, PUBS, STUDENTS, type Section } from "../data";

/* The lens flare of the group's banner, redrawn in CSS so it carries no
 * competing wordmark and stays sharp at any size. */
function Glass() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const y = Math.min(window.scrollY, 900);
        el.style.transform = `translate3d(0, ${y * 0.18}px, 0)`;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div
      ref={ref}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-0 will-change-transform"
    >
      <div
        className="absolute -right-24 -top-40 h-[46rem] w-[46rem] rounded-full opacity-70 blur-[90px]"
        style={{
          background:
            "radial-gradient(circle at 40% 40%, #cfc6fb 0%, rgba(207,198,251,0) 68%)",
        }}
      />
      <div
        className="absolute -bottom-56 right-4 h-[38rem] w-[38rem] rounded-full opacity-60 blur-[90px]"
        style={{
          background:
            "radial-gradient(circle at 50% 50%, #b6e4d9 0%, rgba(182,228,217,0) 66%)",
        }}
      />
      <div
        className="absolute -top-24 left-1/3 h-[26rem] w-[26rem] rounded-full opacity-50 blur-[80px]"
        style={{
          background:
            "radial-gradient(circle at 50% 50%, #f7cfe0 0%, rgba(247,207,224,0) 66%)",
        }}
      />
    </div>
  );
}

export default function Hero({ go }: { go: (s: Section) => void }) {
  const members = FACULTY.length + STUDENTS.length;

  return (
    <section id="home" className="relative overflow-hidden border-b border-rule">
      <Glass />

      {/* The sphere, unwrapped. Fades out under the headline. */}
      <div
        className="pointer-events-none absolute inset-y-0 right-0 z-0 w-full md:w-[62%]"
        style={{
          maskImage:
            "linear-gradient(90deg, transparent 0%, #000 42%, #000 88%, transparent 100%)",
          WebkitMaskImage:
            "linear-gradient(90deg, transparent 0%, #000 42%, #000 88%, transparent 100%)",
        }}
      >
        <PointCloud className="absolute inset-0 h-full w-full opacity-55 md:opacity-100" />
      </div>

      <div className="shell relative z-10 flex min-h-[min(82vh,860px)] flex-col justify-center pb-10 pt-28">
        <div className="max-w-[46rem]">
          <p className="mono rise text-mute">
            {GROUP.name} · {GROUP.institution}
          </p>

          <h1
            className="dsp dsp-xl rise mt-7 text-[clamp(2.2rem,5.6vw,4.4rem)]"
            style={{ animationDelay: "90ms" }}
          >
            Visual Computing
            <br />
            Research Group
          </h1>

          <p
            className="rise mt-8 max-w-[52ch] text-[1.125rem] leading-relaxed text-mute"
            style={{ animationDelay: "180ms" }}
          >
            Research on compression, computer vision and data mining for
            omnidirectional visual content: 360° video and spherical imagery.
          </p>

          <div
            className="rise mt-10 flex flex-wrap items-center gap-x-8 gap-y-4"
            style={{ animationDelay: "260ms" }}
          >
            <button
              onClick={() => go("research")}
              className="mono inline-flex items-center gap-2.5 rounded-[3px] bg-ink px-5 py-3 text-paper transition-opacity hover:opacity-85"
            >
              See the research
              <ArrowDown size={13} />
            </button>
            <button
              onClick={() => go("join")}
              className="mono ulink text-mute transition-colors hover:text-ink"
            >
              Join the group
            </button>
          </div>
        </div>
      </div>

      {/* The hero's footer strip: real counts, and the point plotted above. */}
      <div className="relative border-t border-rule">
        <div className="shell flex flex-wrap items-center gap-x-10 gap-y-3 py-4">
          {[
            [LINES.length, "research lines"],
            [PUBS.length, "publications"],
            [members, "members"],
          ].map(([n, label]) => (
            <span key={label as string} className="flex items-baseline gap-2">
              <span className="num text-[0.95rem] text-ink">
                {String(n).padStart(2, "0")}
              </span>
              <span className="mono text-faint">{label}</span>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
