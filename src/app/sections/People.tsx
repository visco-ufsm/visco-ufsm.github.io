import { ALUMNI, FACULTY, PEOPLE_TEXT, STUDENTS, type Person } from "../data";
import { Collapse, Reveal, SectionHead, Thumb } from "../components/primitives";

const hasLink = (p: Person) => p.link !== "" && p.link !== "#";

function Portrait({ p, size }: { p: Person; size: number }) {
  const Shell = hasLink(p) ? "a" : "span";
  return (
    <Shell
      {...(hasLink(p)
        ? { href: p.link, target: "_blank", rel: "noopener noreferrer" }
        : {})}
      className="group block shrink-0 rounded-full"
      style={{ width: size, height: size }}
      tabIndex={-1}
      aria-hidden="true"
    >
      <span
        className="block h-full w-full overflow-hidden rounded-full bg-rule ring-1 ring-rule transition-all duration-300 group-hover:ring-2 group-hover:ring-iris"
        style={{ boxShadow: "0 0 0 4px var(--color-paper)" }}
      >
        <Thumb
          src={p.photo}
          className="h-full w-full object-cover grayscale transition-all duration-500 group-hover:grayscale-0"
        />
      </span>
    </Shell>
  );
}

function FacultyCard({ p }: { p: Person }) {
  return (
    <div className="flex items-center gap-5 border-b border-rule py-7 sm:border-b-0 sm:py-0">
      <Portrait p={p} size={92} />
      <div className="min-w-0">
        {hasLink(p) ? (
          <a
            href={p.link}
            target="_blank"
            rel="noopener noreferrer"
            className="dsp ulink text-[1.0625rem]"
          >
            {p.name}
          </a>
        ) : (
          <span className="dsp text-[1.0625rem]">{p.name}</span>
        )}
        {/* whitespace-pre-line: a \n written in content.ts becomes a real
            line break here. */}
        <p className="mono mt-2 whitespace-pre-line text-faint">{p.role}</p>
        <p className="mt-1 whitespace-pre-line text-[0.875rem] text-mute">{p.area}</p>
      </div>
    </div>
  );
}

function StudentCard({ p }: { p: Person }) {
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <Portrait p={p} size={78} />
      <div>
        {hasLink(p) ? (
          <a
            href={p.link}
            target="_blank"
            rel="noopener noreferrer"
            className="ulink text-[0.875rem] font-medium leading-snug"
          >
            {p.name}
          </a>
        ) : (
          <span className="text-[0.875rem] font-medium leading-snug">{p.name}</span>
        )}
        <p className="mono mt-1.5 whitespace-pre-line text-[0.625rem] text-faint">
          {p.role}
        </p>
        <p className="mt-1 whitespace-pre-line text-[0.8125rem] leading-snug text-mute">
          {p.area}
        </p>
      </div>
    </div>
  );
}

export default function People() {
  return (
    <section id="people" className="band shell border-b border-rule">
      <SectionHead
        title={PEOPLE_TEXT.title}
        intro={PEOPLE_TEXT.intro}
      />

      <Collapse label={PEOPLE_TEXT.facultyLabel} count={FACULTY.length}>
        <div className="grid gap-x-10 gap-y-8 sm:grid-cols-2 xl:grid-cols-3">
          {FACULTY.map((p) => (
            <Reveal key={p.name}>
              <FacultyCard p={p} />
            </Reveal>
          ))}
        </div>
      </Collapse>

      <Collapse label={PEOPLE_TEXT.studentsLabel} count={STUDENTS.length}>
        <div className="grid grid-cols-2 gap-x-6 gap-y-10 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
          {STUDENTS.map((p, i) => (
            <Reveal key={p.name} delay={i * 45}>
              <StudentCard p={p} />
            </Reveal>
          ))}
        </div>
      </Collapse>

      <Collapse label={PEOPLE_TEXT.alumniLabel} count={ALUMNI.length} defaultOpen={false}>
        <ul className="grid gap-x-10 sm:grid-cols-2 lg:grid-cols-3">
          {ALUMNI.map((a) => (
            <li
              key={a.name}
              className="flex items-baseline justify-between gap-4 border-b border-rule py-3.5"
            >
              <span className="text-[0.9375rem]">{a.name}</span>
              <span className="num text-[0.7rem] text-faint">{a.degree}</span>
            </li>
          ))}
        </ul>
      </Collapse>
    </section>
  );
}
