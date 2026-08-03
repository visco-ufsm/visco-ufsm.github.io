import { ArrowUpRight } from "lucide-react";
import { GROUP, NEWS } from "../data";
import { Reveal, SectionHead, Tag, Thumb } from "../components/primitives";

const FACTS: [string, string][] = [
  ["Institution", "Universidade Federal de Santa Maria"],
  ["Unit", "Departamento de Computação Aplicada"],
  ["Registry", "PRPGP/UFSM"],
  ["Societies", GROUP.affiliations.slice(0, 3).join(" · ")],
];

function NewsCard({ item, i }: { item: (typeof NEWS)[number]; i: number }) {
  const external = item.href?.startsWith("http");

  const body = (
    <>
      <div className="overflow-hidden border border-rule bg-white">
        <Thumb
          src={item.img}
          className="aspect-[16/10] w-full object-cover transition-transform duration-700 group-hover:scale-[1.04]"
        />
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
    <Reveal delay={i * 60}>
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
    </Reveal>
  );
}

export default function About() {
  return (
    <section id="about" className="band shell border-b border-rule">
      <SectionHead eyebrow="About" title="Who we are" />

      <div className="grid gap-12 lg:grid-cols-12">
        <Reveal className="lg:col-span-7">
          <p className="max-w-[54ch] text-[1.25rem] leading-relaxed text-ink">
            The group covers the full pipeline of omnidirectional visual content:
            coding and transmission of 360° video, recognition and segmentation on
            spherical imagery, and quality assessment of panoramic content.
          </p>
          <p className="mt-6 max-w-[58ch] text-[1.0625rem] leading-relaxed text-mute">
            The group publishes in journals and conferences, releases code where
            possible, and supervises students from undergraduate research through the
            M.Sc. and Ph.D.
          </p>
        </Reveal>

        <Reveal delay={80} className="lg:col-span-5">
          <dl className="border-t border-rule">
            {FACTS.map(([k, v]) => (
              <div
                key={k}
                className="grid grid-cols-[7.5rem_1fr] gap-4 border-b border-rule py-4"
              >
                <dt className="mono pt-0.5 text-faint">{k}</dt>
                <dd className="text-[0.9375rem] leading-snug text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </Reveal>
      </div>

      <div className="mt-20">
        <div className="mb-8 flex items-end justify-between gap-6 border-b border-rule pb-4">
          <h3 className="mono text-ink">Recent</h3>
          <a
            href={GROUP.github}
            target="_blank"
            rel="noopener noreferrer"
            className="mono ulink text-faint transition-colors hover:text-ink"
          >
            GitHub <ArrowUpRight size={12} />
          </a>
        </div>
        <div className="grid gap-x-8 gap-y-12 sm:grid-cols-2 lg:grid-cols-3">
          {NEWS.map((n, i) => (
            <NewsCard key={n.text} item={n} i={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
