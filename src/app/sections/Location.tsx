import { ArrowUpRight } from "lucide-react";
import Graticule from "../components/Graticule";
import { GROUP, PLACE } from "../data";
import { Reveal, SectionHead } from "../components/primitives";

export default function Location() {
  return (
    <section id="location" className="band shell border-b border-rule">
      <SectionHead eyebrow="Location" title="Where we are" />

      <div className="grid gap-12 lg:grid-cols-12 lg:gap-16">
        <Reveal className="lg:col-span-5">
          <address className="not-italic">
            <p className="text-[1.0625rem] leading-relaxed text-ink">
              {PLACE.lines.map((l) => (
                <span key={l} className="block">
                  {l}
                </span>
              ))}
            </p>
          </address>

          <dl className="mt-8 border-t border-rule">
            <div className="grid grid-cols-[6rem_1fr] gap-4 border-b border-rule py-4">
              <dt className="mono pt-1 text-faint">Coordinates</dt>
              <dd className="num text-[0.875rem] text-ink">
                {PLACE.latLabel} {PLACE.lonLabel}
              </dd>
            </div>
            <div className="grid grid-cols-[6rem_1fr] gap-4 border-b border-rule py-4">
              <dt className="mono pt-1 text-faint">Email</dt>
              <dd>
                <a href={`mailto:${GROUP.email}`} className="ulink text-iris">
                  {GROUP.email}
                </a>
              </dd>
            </div>
            <div className="grid grid-cols-[6rem_1fr] gap-4 border-b border-rule py-4">
              <dt className="mono pt-1 text-faint">Code</dt>
              <dd>
                <a
                  href={GROUP.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ulink text-jade"
                >
                  github.com/visco-ufsm <ArrowUpRight size={11} />
                </a>
              </dd>
            </div>
          </dl>

          <p className="mt-8 max-w-[46ch] text-[0.9375rem] leading-relaxed text-mute">
            To visit, send an email in advance so someone can meet you at the building
            entrance.
          </p>
        </Reveal>

        <Reveal delay={80} className="lg:col-span-7">
          <figure className="border border-rule bg-white">
            <div className="relative aspect-[16/11] w-full overflow-hidden">
              <Graticule variant="map" className="absolute inset-0 h-full w-full" />
              <span className="num absolute bottom-3 left-3 text-[0.65rem] text-faint">
                rings: 1 km · 2 km
              </span>
              <span className="num absolute right-3 top-3 text-[0.65rem] text-faint">
                {PLACE.latLabel} {PLACE.lonLabel}
              </span>
            </div>
            <figcaption className="flex flex-col items-start gap-2 border-t border-rule px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
              <span className="mono text-faint">
                Centro de Tecnologia · Campus Camobi
              </span>
              <a
                href={PLACE.maps}
                target="_blank"
                rel="noopener noreferrer"
                className="mono ulink text-ink"
              >
                Open in maps <ArrowUpRight size={11} />
              </a>
            </figcaption>
          </figure>
        </Reveal>
      </div>
    </section>
  );
}
