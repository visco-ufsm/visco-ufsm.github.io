import viscoLogo from "@/imports/image-2.png";
import { GROUP, LABELS, PLACE, SECTIONS, type Section } from "../data";

export default function Footer({ go }: { go: (s: Section) => void }) {
  return (
    <footer>
      <div className="spectrum-bar h-[2px]" />
      <div className="shell grid gap-10 py-12 md:grid-cols-[1fr_auto] md:items-start">
        <div>
          <div className="flex items-center gap-3">
            <img
              src={viscoLogo}
              alt=""
              className="h-7 w-7 rounded-full"
              style={{ mixBlendMode: "multiply" }}
            />
            <span className="dsp text-[1rem]">VisCo</span>
            <span className="mono text-faint">{GROUP.full}</span>
          </div>
          <p className="num mt-5 text-[0.75rem] leading-relaxed text-faint">
            {PLACE.lines[1]}, {PLACE.lines[3]}
            <br />
            {PLACE.latLabel} {PLACE.lonLabel}
          </p>
          <div className="mono mt-5 flex flex-wrap gap-x-6 gap-y-2">
            <a href={`mailto:${GROUP.email}`} className="ulink text-mute">
              Email
            </a>
            <a
              href={GROUP.github}
              target="_blank"
              rel="noopener noreferrer"
              className="ulink text-mute"
            >
              GitHub
            </a>
            <a
              href="https://www.ufsm.br"
              target="_blank"
              rel="noopener noreferrer"
              className="ulink text-mute"
            >
              ufsm.br
            </a>
          </div>
        </div>

        <div className="flex flex-col gap-6 md:items-end">
          <nav className="mono flex flex-wrap gap-x-6 gap-y-2 md:justify-end">
            {SECTIONS.map((s) => (
              <button
                key={s}
                onClick={() => go(s)}
                className="text-faint transition-colors hover:text-ink"
              >
                {LABELS[s]}
              </button>
            ))}
          </nav>
          <p className="num text-[0.7rem] text-faint">
            © {new Date().getFullYear()} VisCo · {GROUP.short}
          </p>
        </div>
      </div>
    </footer>
  );
}
