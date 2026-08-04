/* Single source of truth for the site's content.
   Edit here: every section, count and filter reads from this file. */

export const SECTIONS = [
  "home",
  "research",
  "people",
  "publications",
  "location",
  "join",
] as const;

export type Section = (typeof SECTIONS)[number];

export const LABELS: Record<Section, string> = {
  home: "Home",
  research: "Research",
  people: "People",
  publications: "Publications",
  location: "Location",
  join: "Join",
};

export const GROUP = {
  name: "VisCo",
  full: "Visual Computing Research Group",
  institution: "Universidade Federal de Santa Maria",
  short: "UFSM",
  email: "thiago.silveira@ufsm.br",
  github: "https://github.com/visco-ufsm",
};

/* Campus coordinates: the datum the hero graticule and the map both plot. */
export const PLACE = {
  lat: -29.71472,
  lon: -53.71722,
  latLabel: "29°42′53″S",
  lonLabel: "53°43′02″W",
  lines: [
    "Departamento de Computação Aplicada",
    "Centro de Tecnologia (CT)",
    "Av. Roraima 1000, Camobi",
    "Santa Maria – RS, 97105-900, Brasil",
  ],
  maps: "https://www.google.com/maps/search/?api=1&query=-29.71472,-53.71722",
  embed:
    "https://www.google.com/maps?q=-29.71472,-53.71722&z=16&hl=pt-BR&output=embed",
};

/* ── Research lines ──────────────────────────────────────────
   Publications and projects reference these by id, so the two
   sections stay aligned instead of duplicating each other. */
export type LineId = "nvc" | "osa" | "vdm" | "sip" | "cv";

export const LINES: { id: LineId; title: string; desc: string }[] = [
  {
    id: "nvc",
    title: "Neural video compression",
    desc: "End-to-end, low-complexity coding of 360° omnidirectional video with deep neural networks, for efficient immersive streaming.",
  },
  {
    id: "cv",
    title: "Computer vision",
    desc: "Pattern recognition and scene understanding for equirectangular and spherical imagery.",
  },
  {
    id: "osa",
    title: "Omnidirectional scene analysis",
    desc: "Detection and segmentation adapted to the geometric distortion of 360° projection.",
  },
  {
    id: "sip",
    title: "Signal and image processing",
    desc: "Restoration, super-resolution and perceptual quality assessment of panoramic and spherical content.",
  },
  {
    id: "vdm",
    title: "Visual data mining",
    desc: "Knowledge extraction from high-dimensional visual datasets, including panoramic video and immersive scenes.",
  },
];

export const LINE_TITLE: Record<LineId, string> = Object.fromEntries(
  LINES.map((l) => [l.id, l.title]),
) as Record<LineId, string>;

/* ── Projects ────────────────────────────────────────────── */
export type Project = {
  title: string;
  year: string;
  desc: string;
  line: LineId;
  tags: string[];
  active: boolean;
  code: string | null;
};

export const PROJECTS: Project[] = [
  {
    title: "Neural compression of 360° videos",
    year: "2023–25",
    desc: "An end-to-end pipeline for low-complexity omnidirectional video coding, targeting efficient immersive streaming.",
    line: "nvc",
    tags: ["Deep learning", "Video coding", "360°"],
    active: true,
    code: "https://github.com/visco-ufsm",
  },
  {
    title: "Pattern recognition in 360° images",
    year: "2024–26",
    desc: "Detection and segmentation robust to the spherical geometry of omnidirectional cameras.",
    line: "cv",
    tags: ["CNN", "Segmentation"],
    active: true,
    code: null,
  },
  {
    title: "Visual mining of complex datasets",
    year: "2024–25",
    desc: "Knowledge extraction from high-dimensional visual data, including panoramic video and immersive scenes.",
    line: "vdm",
    tags: ["Data mining", "ML"],
    active: false,
    code: "https://github.com/visco-ufsm",
  },
];

/* ── Publications ────────────────────────────────────────── */
export type PubType = "Journal" | "Conference";

export type Pub = {
  year: number;
  title: string;
  authors: string;
  venue: string;
  type: PubType;
  line: LineId;
  pdf: string | null;
  doi: string | null;
  code: string | null;
};

export const PUBS: Pub[] = [
  {
    year: 2025,
    title: "Low-complexity end-to-end neural compression for 360° videos",
    authors: "Silveira, T. L. T. et al.",
    venue: "IEEE Transactions on Image Processing",
    type: "Journal",
    line: "nvc",
    pdf: "#",
    doi: "#",
    code: "#",
  },
  {
    year: 2024,
    title: "Omnidirectional scene understanding via spherical convolutions",
    authors: "Silveira, T. L. T. et al.",
    venue: "SIBGRAPI 2024",
    type: "Conference",
    line: "osa",
    pdf: "#",
    doi: "#",
    code: "#",
  },
  {
    year: 2024,
    title: "Visual data mining for multidimensional 360° datasets",
    authors: "Silveira, T. L. T. et al.",
    venue: "IJCNN 2024",
    type: "Conference",
    line: "vdm",
    pdf: "#",
    doi: "#",
    code: null,
  },
  {
    year: 2023,
    title: "Perceptual quality metrics for equirectangular video compression",
    authors: "Silveira, T. L. T. et al.",
    venue: "IEEE Signal Processing Letters",
    type: "Journal",
    line: "sip",
    pdf: "#",
    doi: "#",
    code: "#",
  },
  {
    year: 2022,
    title: "Deep learning approaches for 360° image super-resolution",
    authors: "Silveira, T. L. T. et al.",
    venue: "SIBGRAPI 2022",
    type: "Conference",
    line: "sip",
    pdf: "#",
    doi: "#",
    code: null,
  },
];

/* ── People ──────────────────────────────────────────────── */
export type Person = {
  name: string;
  role: string;
  area: string;
  photo: string;
  link: string;
};

export const FACULTY: Person[] = [
  {
    name: "Thiago L. T. da Silveira",
    role: "Principal investigator",
    area: "360° video · computer vision",
    photo:
      "https://images.unsplash.com/photo-1568602471122-7832951cc4c5?w=400&h=400&fit=crop&auto=format",
    link: "https://lattes.cnpq.br",
  },
];

export const STUDENTS: Person[] = [
  {
    name: "Student A",
    role: "M.Sc.",
    area: "Neural compression",
    photo:
      "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&auto=format",
    link: "https://github.com",
  },
  {
    name: "Student B",
    role: "M.Sc.",
    area: "Omnidirectional vision",
    photo:
      "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=400&fit=crop&auto=format",
    link: "https://github.com",
  },
  {
    name: "Student C",
    role: "Undergraduate",
    area: "Image processing",
    photo:
      "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&auto=format",
    link: "https://github.com",
  },
  {
    name: "Student D",
    role: "Undergraduate",
    area: "Visual data mining",
    photo:
      "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=400&fit=crop&auto=format",
    link: "https://github.com",
  },
  {
    name: "Student E",
    role: "Undergraduate",
    area: "360° scene analysis",
    photo:
      "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop&auto=format",
    link: "https://github.com",
  },
];

/* Kept on the page on purpose: a record of everyone who passed through. */
export const ALUMNI = [
  { name: "Former Student A", degree: "M.Sc. 2023", work: null as string | null },
  { name: "Former Student B", degree: "B.Sc. 2022", work: null as string | null },
];

/* ── News ────────────────────────────────────────────────── */
export type NewsTag =
  | "Publication"
  | "Event"
  | "Project"
  | "Resource"
  | "Opportunity";

export const NEWS: {
  date: string;
  tag: NewsTag;
  text: string;
  img: string;
  href: string | null;
}[] = [
  {
    date: "Jun 2025",
    tag: "Publication",
    text: "Neural compression of 360° video accepted at IEEE Transactions on Image Processing.",
    img: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&h=500&fit=crop&auto=format",
    href: "#publications",
  },
  {
    date: "Jun 2025",
    tag: "Event",
    text: "VisCo presents omnidirectional computer vision at SBC 2025, Florianópolis.",
    img: "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=800&h=500&fit=crop&auto=format",
    href: null,
  },
  {
    date: "May 2025",
    tag: "Project",
    text: "New project starts: 360° scene analysis for autonomous vehicles.",
    img: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&h=500&fit=crop&auto=format",
    href: "#research",
  },
  {
    date: "Mar 2025",
    tag: "Resource",
    text: "Our 360° benchmark dataset is now open access on GitHub.",
    img: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&h=500&fit=crop&auto=format",
    href: GROUP.github,
  },
  {
    date: "Feb 2025",
    tag: "Opportunity",
    text: "M.Sc. scholarship open in neural video coding, funded by CAPES/CNPq.",
    img: "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=800&h=500&fit=crop&auto=format",
    href: "#join",
  },
  {
    date: "Dec 2024",
    tag: "Publication",
    text: "Spherical convolutions for scene understanding presented at SIBGRAPI 2024.",
    img: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=500&fit=crop&auto=format",
    href: "#publications",
  },
];

export const TAG_COLOR: Record<NewsTag | PubType, string> = {
  Publication: "var(--color-iris)",
  Journal: "var(--color-iris)",
  Conference: "var(--color-jade)",
  Event: "var(--color-jade)",
  Project: "var(--color-blush)",
  Resource: "var(--color-jade)",
  Opportunity: "var(--color-blush)",
};

export const pubsInLine = (id: LineId) => PUBS.filter((p) => p.line === id).length;
