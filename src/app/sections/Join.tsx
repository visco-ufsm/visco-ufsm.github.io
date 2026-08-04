import { ArrowUpRight } from "lucide-react";
import { GROUP } from "../data";
import { Reveal, SectionHead } from "../components/primitives";

const COLUMNS = [
  {
    label: "Students",
    body: [
      "The group supervises M.Sc. and Ph.D. candidates and undergraduate researchers in Computer Science and Electrical Engineering.",
      "A background in image processing, computer vision or machine learning is helpful.",
      "CAPES and CNPq scholarships are available when positions open. Current openings are listed under Recent, in the home section.",
    ],
  },
  {
    label: "Collaborators",
    body: [
      "VisCo co-advises students, publishes jointly, and takes part in funded research with academic groups and industry partners.",
      "The group is affiliated with SBC, SBMAC and IEEE, and registered with PRPGP/UFSM.",
      "Collaborations are open in optics, coding, perception and robotics applied to omnidirectional imaging.",
    ],
  },
];

export default function Join() {
  return (
    <section id="join" className="band shell">
      <SectionHead title="How to join" />

      <div className="grid gap-x-16 gap-y-12 md:grid-cols-2">
        {COLUMNS.map((c, i) => (
          <Reveal key={c.label} delay={i * 80}>
            <div className="spectrum-bar mb-5 h-px w-8" />
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
        <div className="mt-16 flex flex-col items-start gap-6 border-t border-rule pt-10 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-[46ch] text-[0.9375rem] leading-relaxed text-mute">
            Write to Prof. Thiago Silveira with a short description of your background
            and research interests.
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
