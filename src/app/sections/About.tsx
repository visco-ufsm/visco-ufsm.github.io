import NewsCarousel from "../components/NewsCarousel";
import { Reveal, SectionHead } from "../components/primitives";

export default function About() {
  return (
    <section id="about" className="band shell border-b border-rule">
      <SectionHead title="Who we are" />

      <div className="grid gap-12 lg:grid-cols-12 lg:gap-16">
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
          <p className="mt-6 max-w-[58ch] text-[1.0625rem] leading-relaxed text-mute">
            VisCo works at the Departamento de Computação Aplicada, is registered with
            PRPGP/UFSM, and is affiliated with SBC, SBMAC and IEEE.
          </p>
        </Reveal>

        <Reveal delay={80} className="lg:col-span-5">
          <NewsCarousel />
        </Reveal>
      </div>
    </section>
  );
}
