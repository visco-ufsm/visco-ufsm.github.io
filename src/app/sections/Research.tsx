import { ArrowUpRight } from "lucide-react";
import {
  LINES,
  PROJECTS,
  RESEARCH,
  pubsInLine,
  type LineId,
  type Project,
} from "../data";
import { Collapse, Reveal, SectionHead } from "../components/primitives";

function ProjectRow({ p }: { p: Project }) {
  return (
    <article className="grid gap-3 border-b border-rule py-6 sm:grid-cols-[7rem_1fr] sm:gap-8">
      <div className="num pt-1 text-[0.75rem] text-faint">{p.year}</div>
      <div>
        <div className="flex items-start justify-between gap-6">
          <h4 className="dsp text-[1.0625rem]">{p.title}</h4>
          {p.code && (
            <a
              href={p.code}
              target="_blank"
              rel="noopener noreferrer"
              className="mono ulink shrink-0 text-jade"
            >
              Code <ArrowUpRight size={11} />
            </a>
          )}
        </div>
        <p className="mt-2 max-w-[70ch] text-[0.9375rem] leading-relaxed text-mute">
          {p.desc}
        </p>
        <div className="mono mt-3 flex flex-wrap gap-x-4 gap-y-1 text-faint">
          {p.tags.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      </div>
    </article>
  );
}

export default function Research({
  onPickLine,
}: {
  onPickLine: (id: LineId) => void;
}) {
  const active = PROJECTS.filter((p) => p.active);
  const done = PROJECTS.filter((p) => !p.active);

  return (
    <section id="research" className="band shell border-b border-rule">
      <SectionHead
        title={RESEARCH.title}
        intro={RESEARCH.intro}
      />

      <div className="border-t border-rule">
        {LINES.map((l, i) => {
          const n = pubsInLine(l.id);
          const projects = PROJECTS.filter((p) => p.lines.includes(l.id)).length;
          const row = (
            <>
              <h3 className="dsp flex items-center gap-3 text-[1.25rem] md:text-[1.375rem]">
                <span className="spectrum-bar h-px w-0 shrink-0 transition-all duration-500 group-hover:w-6" />
                {l.title}
              </h3>
              <p className="text-[0.9375rem] leading-relaxed text-mute">{l.desc}</p>
              <span className="mono flex items-center gap-2 text-faint transition-colors group-hover:text-ink md:justify-end">
                {n > 0 ? (
                  <>
                    {n} {n === 1 ? "paper" : "papers"}
                    <ArrowUpRight size={12} />
                  </>
                ) : (
                  `${projects} ${projects === 1 ? "project" : "projects"} running`
                )}
              </span>
            </>
          );
          const shape =
            "group grid w-full grid-cols-1 items-start gap-3 border-b border-rule py-7 text-left md:grid-cols-[minmax(0,4fr)_minmax(0,6fr)_minmax(0,2fr)] md:gap-10";

          return (
            <Reveal key={l.id} delay={i * 50}>
              {n > 0 ? (
                <button
                  onClick={() => onPickLine(l.id)}
                  className={`${shape} transition-colors hover:bg-ink/[0.02]`}
                >
                  {row}
                </button>
              ) : (
                <div className={shape}>{row}</div>
              )}
            </Reveal>
          );
        })}
      </div>

      <div className="mt-12">
        <Collapse label={RESEARCH.activeLabel} count={active.length}>
          <div className="border-t border-rule">
            {active.map((p) => (
              <ProjectRow key={p.title} p={p} />
            ))}
          </div>
        </Collapse>

        {done.length > 0 && (
          <Collapse label={RESEARCH.concludedLabel} count={done.length} defaultOpen={false}>
            <div className="border-t border-rule">
              {done.map((p) => (
                <ProjectRow key={p.title} p={p} />
              ))}
            </div>
          </Collapse>
        )}
      </div>
    </section>
  );
}
