import { useState } from "react";
import { Menu, X } from "lucide-react";
import viscoLogo from "@/imports/image-2.png";
import { LABELS, SECTIONS, type Section } from "../data";

/* The only place the logo appears. The hero shows the same sphere unwrapped,
 * so the mark is never duplicated on screen. */
export default function Nav({
  active,
  go,
}: {
  active: Section;
  go: (s: Section) => void;
}) {
  const [open, setOpen] = useState(false);

  const jump = (s: Section) => {
    setOpen(false);
    go(s);
  };

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-rule/80 bg-paper/80 backdrop-blur-xl">
      <nav className="shell flex h-14 items-center justify-between">
        <button
          onClick={() => jump("home")}
          className="flex items-center gap-2.5"
          aria-label="VisCo, back to top"
        >
          <img
            src={viscoLogo}
            alt=""
            className="h-6 w-6 rounded-full"
            style={{ mixBlendMode: "multiply" }}
          />
          <span className="dsp text-[0.95rem] leading-none">VisCo</span>
          <span className="mono hidden text-[0.6rem] text-faint sm:inline">UFSM</span>
        </button>

        <ul className="hidden items-center gap-7 md:flex">
          {SECTIONS.map((s) => (
            <li key={s}>
              <button
                onClick={() => jump(s)}
                aria-current={active === s ? "true" : undefined}
                className={`mono relative pb-1.5 transition-colors ${
                  active === s ? "text-ink" : "text-faint hover:text-mute"
                }`}
              >
                {LABELS[s]}
                <span
                  className={`spectrum-bar absolute inset-x-0 bottom-0 h-[2px] origin-left transition-transform duration-300 ${
                    active === s ? "scale-x-100" : "scale-x-0"
                  }`}
                />
              </button>
            </li>
          ))}
        </ul>

        <button
          onClick={() => setOpen((o) => !o)}
          className="text-ink md:hidden"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
        >
          {open ? <X size={18} /> : <Menu size={18} />}
        </button>
      </nav>

      {open && (
        <ul className="border-t border-rule bg-paper md:hidden">
          {SECTIONS.map((s) => (
            <li key={s}>
              <button
                onClick={() => jump(s)}
                className={`mono flex w-full items-center gap-3 border-b border-rule px-6 py-4 text-left ${
                  active === s ? "text-ink" : "text-faint"
                }`}
              >
                {active === s && <span className="spectrum-bar h-px w-4" />}
                {LABELS[s]}
              </button>
            </li>
          ))}
        </ul>
      )}
    </header>
  );
}
