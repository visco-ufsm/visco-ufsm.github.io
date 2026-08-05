import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import {
  LINES,
  LINE_TITLE,
  PUBLICATIONS_TEXT,
  PUBS,
  type LineId,
  type Pub,
  type PubType,
} from "../data";
import { SectionHead, Tag } from "../components/primitives";

const TYPES = ["All", "Journal", "Conference"] as const;

function Entry({ p }: { p: Pub }) {
  const links: [string, string | null][] = [
    ["PDF", p.pdf],
    ["DOI", p.doi],
    ["Code", p.code],
  ];
  return (
    <article className="border-b border-rule py-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-8">
        <h4 className="dsp max-w-[62ch] text-[1.0625rem] leading-snug">{p.title}</h4>
        <div className="mono flex shrink-0 gap-4">
          {links.map(([label, href]) =>
            href ? (
              <a
                key={label}
                href={href}
                className={`ulink ${label === "Code" ? "text-jade" : "text-ink"}`}
              >
                {label}
              </a>
            ) : null,
          )}
        </div>
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[0.8125rem] text-mute">
        <span>{p.authors}</span>
        <span className="italic">{p.venue}</span>
        <Tag type={p.type} />
        <span className="mono text-faint">{LINE_TITLE[p.line]}</span>
      </div>
    </article>
  );
}

export default function Publications({
  line,
  setLine,
}: {
  line: LineId | null;
  setLine: (l: LineId | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState<(typeof TYPES)[number]>("All");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return PUBS.filter((p) => {
      if (type !== "All" && p.type !== (type as PubType)) return false;
      if (line && p.line !== line) return false;
      if (!q) return true;
      return (
        p.title.toLowerCase().includes(q) ||
        p.venue.toLowerCase().includes(q) ||
        p.authors.toLowerCase().includes(q)
      );
    });
  }, [query, type, line]);

  const years = [...new Set(filtered.map((p) => p.year))].sort((a, b) => b - a);
  const dirty = query !== "" || type !== "All" || line !== null;

  const clear = () => {
    setQuery("");
    setType("All");
    setLine(null);
  };

  return (
    <section id="publications" className="band shell border-b border-rule">
      <SectionHead
        title={PUBLICATIONS_TEXT.title}
        right={
          <label className="flex items-center gap-2.5 border-b border-rule pb-2 sm:w-64">
            <Search size={14} className="shrink-0 text-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={PUBLICATIONS_TEXT.searchPlaceholder}
              className="w-full bg-transparent text-[0.875rem] text-ink placeholder:text-faint focus:outline-none"
            />
          </label>
        }
      />

      <div className="mb-2 flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-rule pb-5">
        <div className="mono flex gap-4">
          {TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setType(t)}
              className={`relative pb-1.5 transition-colors ${
                type === t ? "text-ink" : "text-faint hover:text-mute"
              }`}
            >
              {t}
              <span
                className={`spectrum-bar absolute inset-x-0 bottom-0 h-px origin-left transition-transform duration-300 ${
                  type === t ? "scale-x-100" : "scale-x-0"
                }`}
              />
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          {LINES.map((l) => {
            const on = line === l.id;
            return (
              <button
                key={l.id}
                onClick={() => setLine(on ? null : l.id)}
                aria-pressed={on}
                className={`mono border px-2.5 py-1.5 transition-colors ${
                  on
                    ? "border-ink text-ink"
                    : "border-rule text-faint hover:border-mute hover:text-mute"
                }`}
              >
                {l.title}
              </button>
            );
          })}
        </div>

        {dirty && (
          <button
            onClick={clear}
            className="mono ml-auto inline-flex items-center gap-1.5 text-iris"
          >
            <X size={12} /> Clear
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <p className="py-12 text-[0.9375rem] text-mute">
          Nothing matches those filters.{" "}
          <button onClick={clear} className="ulink text-iris">
            Show all {PUBS.length} publications
          </button>
        </p>
      ) : (
        /* One block per year that actually has papers, all of them open. */
        <div className="mt-8">
          {years.map((year) => {
            const inYear = filtered.filter((p) => p.year === year);
            return (
              <div
                key={year}
                className="grid gap-2 border-t border-rule pt-6 first:border-t-0 md:grid-cols-[7rem_1fr] md:gap-8"
              >
                <div className="md:sticky md:top-20 md:self-start">
                  <p className="num text-[1.25rem] leading-none text-ink">{year}</p>
                  <p className="mono mt-2 text-faint">
                    {inYear.length} {inYear.length === 1 ? "paper" : "papers"}
                  </p>
                </div>
                <div>
                  {inYear.map((p) => (
                    <Entry key={p.title} p={p} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
