import { useCallback, useEffect, useState } from "react";
import Nav from "./components/Nav";
import Hero from "./sections/Hero";
import About from "./sections/About";
import Research from "./sections/Research";
import People from "./sections/People";
import Publications from "./sections/Publications";
import Location from "./sections/Location";
import Join from "./sections/Join";
import Footer from "./sections/Footer";
import { SECTIONS, type LineId, type Section } from "./data";

const NAV_H = 56;

export default function App() {
  const [active, setActive] = useState<Section>("home");
  const [line, setLine] = useState<LineId | null>(null);

  const go = useCallback((s: Section) => {
    const el = document.getElementById(s);
    if (!el) return;
    window.scrollTo({
      top: el.getBoundingClientRect().top + window.scrollY - NAV_H,
      behavior: "smooth",
    });
  }, []);

  /* Picking a research line filters the publication list and takes you there.
     The two sections describe the same work from different ends. */
  const pickLine = useCallback(
    (id: LineId) => {
      setLine(id);
      go("publications");
    },
    [go],
  );

  useEffect(() => {
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        for (const s of [...SECTIONS].reverse()) {
          const el = document.getElementById(s);
          if (el && el.getBoundingClientRect().top <= NAV_H + 40) {
            setActive(s);
            return;
          }
        }
        setActive("home");
      });
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div className="min-h-screen bg-paper text-ink">
      <a
        href="#home"
        className="mono sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:bg-ink focus:px-4 focus:py-2 focus:text-paper"
      >
        Skip to content
      </a>

      <Nav active={active} go={go} />

      <main>
        <Hero go={go} />
        <About />
        <Research onPickLine={pickLine} />
        <People />
        <Publications line={line} setLine={setLine} />
        <Location />
        <Join />
      </main>

      <Footer go={go} />
    </div>
  );
}
