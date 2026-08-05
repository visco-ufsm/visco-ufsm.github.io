import { ArrowUpRight } from "lucide-react";
import { GROUP, JOIN } from "../data";
import { Reveal, SectionHead } from "../components/primitives";

export default function Join() {
  return (
    <section id="join" className="band shell">
      <SectionHead title={JOIN.title} />

      <div className="grid gap-x-16 gap-y-12 md:grid-cols-2">
        {JOIN.columns.map((c, i) => (
          <Reveal key={c.label} delay={i * 80}>
            <h3 className="mono text-ink">{c.label}</h3>
            <div className="mt-4 space-y-4">
              {c.body.map((p) => (
                <p key={p} className="max-w-[52ch] leading-relaxed text-mute">
                  {p}
                </p>
              ))}
            </div>
          </Reveal>
        ))}
      </div>

      <Reveal>
        <div className="mt-12 flex flex-col items-start gap-6 border-t border-rule pt-8 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-[46ch] text-[0.9375rem] leading-relaxed text-mute">
            {JOIN.ctaNote}
          </p>
          <a
            href={`mailto:${GROUP.email}?subject=VisCo`}
            className="mono inline-flex shrink-0 items-center gap-2.5 rounded-[3px] bg-ink px-5 py-3 text-paper transition-opacity hover:opacity-85"
            style={{ textTransform: "none" }}
          >
            {GROUP.email}
            <ArrowUpRight size={13} />
          </a>
        </div>
      </Reveal>
    </section>
  );
}
