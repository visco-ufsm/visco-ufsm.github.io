import NewsCarousel from "../components/NewsCarousel";
import { ABOUT } from "../data";
import { Reveal, SectionHead } from "../components/primitives";

export default function About() {
  return (
    <section id="about" className="band shell border-b border-rule">
      <SectionHead title={ABOUT.title} />

      {/* items-stretch + h-full down the carousel: its bottom edge ends level
          with the last paragraph, the image absorbing the height difference. */}
      <div className="grid items-stretch gap-12 lg:grid-cols-12 lg:gap-16">
        <Reveal className="lg:col-span-7">
          {/* The first paragraph leads, the rest follow at reading size. */}
          {ABOUT.paragraphs.map((text, i) =>
            i === 0 ? (
              <p
                key={text}
                className="max-w-[54ch] text-[1.25rem] leading-relaxed text-ink"
              >
                {text}
              </p>
            ) : (
              <p
                key={text}
                className="mt-6 max-w-[58ch] text-[1.0625rem] leading-relaxed text-mute"
              >
                {text}
              </p>
            ),
          )}
        </Reveal>

        <Reveal delay={80} className="h-full lg:col-span-5">
          <NewsCarousel />
        </Reveal>
      </div>
    </section>
  );
}
