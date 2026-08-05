/* Derived values and presentation constants.
 *
 * Nothing editable lives here — all content is in `src/content.ts`.
 * Components import from this module so they get both the content and the
 * values computed from it. */

import { LINES, PUBS, type LineId, type NewsTag, type PubType } from "@/content";

export * from "@/content";

/** Research line titles keyed by id, for labelling publications. */
export const LINE_TITLE: Record<LineId, string> = Object.fromEntries(
  LINES.map((l) => [l.id, l.title]),
) as Record<LineId, string>;

/** How many publications a research line has produced. */
export const pubsInLine = (id: LineId) => PUBS.filter((p) => p.line === id).length;

/** Dot colour for news tags and publication types. */
export const TAG_COLOR: Record<NewsTag | PubType, string> = {
  Publication: "var(--color-iris)",
  Journal: "var(--color-iris)",
  Conference: "var(--color-jade)",
  Event: "var(--color-jade)",
  Project: "var(--color-blush)",
  Resource: "var(--color-jade)",
  Opportunity: "var(--color-blush)",
};
