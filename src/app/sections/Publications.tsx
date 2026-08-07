import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import {
  LINES,
  OLDER_AFTER,
  PUBLICATIONS_TEXT,
  PUBS,
  TAG_COLOR,
  type LineId,
  type Pub,
  type PubType,
} from "../data";
import { Collapse, SectionHead } from "../components/primitives";

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
        {/* The dot carries the venue type implicitly: iris = journal,
            jade = conference. The filter buttons above are the legend. */}
        <h4 className="dsp max-w-[62ch] text-[1.0625rem] leading-snug">
          <span
            title={p.type}
            className="mb-0.5 mr-2.5 inline-block h-[7px] w-[7px] rounded-full align-middle"
            style={{ background: TAG_COLOR[p.type] }}
          />
          <span className="sr-only">{p.type}: </span>
          {p.title}
        </h4>
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
        {p.collab && (
          <span className="mono inline-flex items-center gap-1.5 text-blush">
            <span className="h-[5px] w-[5px] rounded-full border border-blush" />
            Collaboration
          </span>
        )}
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
      if (line && !p.lines.includes(line)) return false;
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

  /* Anything past OLDER_AFTER years drops out of the main list into a fold that
     starts closed, so the list stays short as the group publishes more. */
  const cutoff = new Date().getFullYear() - OLDER_AFTER;
  const recentYears = years.filter((y) => y >= cutoff);
  const olderYears = years.filter((y) => y < cutoff);
  const olderCount = filtered.filter((p) => p.year < cutoff).length;

  const clear = () => {
    setQuery("");
    setType("All");
    setLine(null);
  };

  /* One block per year, each with its own count. */
  const YearGroups = ({ list }: { list: number[] }) => (
    <>
      {list.map((year) => {
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
    </>
  );

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
              className={`relative inline-flex items-center gap-1.5 pb-1.5 transition-colors ${
                type === t ? "text-ink" : "text-faint hover:text-mute"
              }`}
            >
              {/* Doubles as the legend for the dot shown on each paper. */}
              {t !== "All" && (
                <span
                  className="h-[6px] w-[6px] rounded-full"
                  style={{ background: TAG_COLOR[t] }}
                />
              )}
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
        <div className="mt-8">
          <YearGroups list={recentYears} />

          {olderYears.length > 0 && (
            <div className={recentYears.length > 0 ? "mt-10" : undefined}>
              <Collapse
                label={PUBLICATIONS_TEXT.olderLabel}
                count={olderCount}
                defaultOpen={false}
              >
                <YearGroups list={olderYears} />
              </Collapse>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
