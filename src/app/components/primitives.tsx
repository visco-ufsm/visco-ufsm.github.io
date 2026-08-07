import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { TAG_COLOR, type NewsTag, type PubType } from "../data";

/** Reveals its children once, on first scroll into view. */
export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Without IntersectionObserver the content must never stay hidden.
  const [seen, setSeen] = useState(
    () => typeof IntersectionObserver === "undefined",
  );

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`reveal ${seen ? "in" : ""} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/** Image that degrades to an on-brand tint instead of a broken-image icon. */
export function Thumb({
  src,
  className = "",
}: {
  src: string;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  if (failed || !src)
    return (
      <div
        className={className}
        style={{
          background: "linear-gradient(135deg,#ece9fa 0%,#e2f1ec 55%,#f7e9f0 100%)",
        }}
      />
    );
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      className={className}
      onError={() => setFailed(true)}
    />
  );
}

/** A colour-coded label: a dot and a word, never a filled chip. */
export function Tag({ type }: { type: NewsTag | PubType }) {
  return (
    <span className="mono inline-flex items-center gap-1.5 text-ink/70">
      <span
        className="h-[5px] w-[5px] rounded-full"
        style={{ background: TAG_COLOR[type] }}
      />
      {type}
    </span>
  );
}

/* No eyebrow label: the section name is already in the menu, and repeating it
   in the running text only adds noise. */
export function SectionHead({
  title,
  intro,
  right,
}: {
  title: string;
  intro?: string;
  right?: ReactNode;
}) {
  return (
    <header className="mb-7">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <h2 className="dsp text-[clamp(1.6rem,3vw,2.35rem)]">{title}</h2>
        {right}
      </div>
      {intro && (
        <p className="mt-4 max-w-[64ch] text-[0.95rem] leading-relaxed text-mute">
          {intro}
        </p>
      )}
    </header>
  );
}

/** Sub-section that folds away older or less relevant material. */
export function Collapse({
  label,
  count,
  children,
  defaultOpen = true,
}: {
  label: string;
  count?: number;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="mt-9 first:mt-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="group flex w-full items-center gap-3 border-b border-rule pb-3 text-left"
      >
        <span className={`mono ${open ? "text-ink" : "text-faint"} transition-colors`}>
          {label}
        </span>
        {count !== undefined && (
          <span className="num text-[0.7rem] text-faint">{count}</span>
        )}
        <span className="ml-auto text-faint transition-colors group-hover:text-ink">
          <ChevronDown
            size={13}
            className={`transition-transform duration-300 ${open ? "rotate-180" : ""}`}
          />
        </span>
      </button>
      {open && <div className="pt-5">{children}</div>}
    </section>
  );
}
