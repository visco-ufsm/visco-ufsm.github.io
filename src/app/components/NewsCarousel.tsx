import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, ChevronLeft, ChevronRight } from "lucide-react";
import { NEWS } from "../data";
import { Tag, Thumb } from "./primitives";

const INTERVAL = 6000;

export default function NewsCarousel() {
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const total = NEWS.length;
  const timer = useRef<number | null>(null);

  const go = useCallback(
    (n: number) => setIdx((i) => (i + n + total) % total),
    [total],
  );

  useEffect(() => {
    if (paused) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    timer.current = window.setInterval(() => go(1), INTERVAL);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [go, paused]);

  const item = NEWS[idx];
  const external = item.href?.startsWith("http");

  const body = (
    <>
      <div className="overflow-hidden border border-rule bg-white">
        <Thumb src={item.img} className="aspect-[16/10] w-full object-cover" />
      </div>
      <div className="mt-4 flex items-center gap-3">
        <span className="num text-[0.7rem] text-faint">{item.date}</span>
        <Tag type={item.tag} />
        {item.href && (
          <ArrowUpRight
            size={12}
            className="ml-auto text-faint transition-colors group-hover:text-ink"
          />
        )}
      </div>
      <p className="mt-2 text-[0.9375rem] leading-snug text-ink">{item.text}</p>
    </>
  );

  return (
    <section
      aria-label="Recent news"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocusCapture={() => setPaused(true)}
      onBlurCapture={() => setPaused(false)}
    >
      <div className="mb-4 flex items-center justify-between gap-4 border-b border-rule pb-3">
        <h3 className="mono text-ink">Recent</h3>
        <div className="flex items-center gap-3">
          <button
            onClick={() => go(-1)}
            aria-label="Previous item"
            className="text-faint transition-colors hover:text-ink"
          >
            <ChevronLeft size={14} />
          </button>
          <div className="flex gap-1.5">
            {NEWS.map((n, i) => (
              <button
                key={n.text}
                onClick={() => setIdx(i)}
                aria-label={`Item ${i + 1} of ${total}`}
                aria-current={i === idx}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === idx ? "spectrum-bar w-5" : "w-1.5 bg-rule hover:bg-faint"
                }`}
              />
            ))}
          </div>
          <button
            onClick={() => go(1)}
            aria-label="Next item"
            className="text-faint transition-colors hover:text-ink"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      <div key={idx} style={{ animation: "fadeIn .45s ease both" }}>
        {item.href ? (
          <a
            href={item.href}
            {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
            className="group block"
          >
            {body}
          </a>
        ) : (
          <div className="group">{body}</div>
        )}
      </div>

      <style>{`@keyframes fadeIn{from{opacity:0;transform:translateX(6px)}to{opacity:1;transform:none}}
        @media (prefers-reduced-motion: reduce){[style*="fadeIn"]{animation:none !important}}`}</style>
    </section>
  );
}
