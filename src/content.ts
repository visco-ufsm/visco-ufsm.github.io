/* ============================================================================
 * CONTEÚDO DO SITE — edite tudo por aqui
 * ============================================================================
 *
 * Este é o único arquivo que precisa ser alterado para atualizar o site:
 * textos, pessoas, publicações, projetos, notícias e localização.
 * Nenhum componente precisa ser tocado.
 *
 * Regras para não quebrar nada:
 *   - Mantenha as aspas e as vírgulas exatamente como estão.
 *   - Aspas dentro de um texto precisam de barra invertida: "o \"cubo\"".
 *   - Campos marcados como `null` significam "não existe"; para preencher,
 *     troque null por um texto entre aspas.
 *   - Os identificadores das linhas de pesquisa (LineId) ligam publicações e
 *     projetos às linhas. Se criar uma linha nova, adicione o id na lista
 *     LineId logo abaixo.
 *
 * Depois de editar: `pnpm build` (ou só commitar — o GitHub Actions publica).
 * ========================================================================== */

/* ── Identidade do grupo ─────────────────────────────────────────────────── */

export const GROUP = {
  name: "VisCo",
  full: "Visual Computing Research Group",
  institution: "Federal University of Santa Maria",
  short: "UFSM",
  email: "thiago.silveira@ufsm.br",
  github: "https://github.com/visco-ufsm",
  university: "https://www.ufsm.br",
};

/* ── Localização ─────────────────────────────────────────────────────────────
 * Para mudar o ponto do mapa, altere apenas `lat` e `lon`. O link e o mapa
 * embutido do Google Maps são montados a partir deles.
 * Pegue as coordenadas clicando com o botão direito no Google Maps.
 * `latLabel` e `lonLabel` são só o texto exibido em graus. */

export const PLACE = {
  lat: -29.712854,
  lon: -53.717101,
  zoom: 16,
  latLabel: "29°42′46.3″S",
  lonLabel: "53°43′01.6″W",
  lines: [
    "Department of Applied Computing",
    "Technology Center (CT)",
    "Ave. Roraima 1000, Camobi",
    "Santa Maria – RS, 97105-900, Brazil",
  ],
};

export const MAPS_EMBED = `https://www.google.com/maps?q=${PLACE.lat},${PLACE.lon}&z=${PLACE.zoom}&hl=pt-BR&output=embed`;
export const MAPS_LINK = `https://www.google.com/maps/search/?api=1&query=${PLACE.lat},${PLACE.lon}`;

/* ── Menu ────────────────────────────────────────────────────────────────────
 * A ordem aqui é a ordem do menu e da página. */

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

/* ── Topo da página ──────────────────────────────────────────────────────── */

export const HERO = {
  // Cada item é uma linha do título. Duas linhas funcionam melhor.
  headline: ["Visual Computing", "Research Group"],
  lead: "Advancing computational methods for signal processing, visual computing, and artificial intelligence.",
  primaryAction: "Explore Research",
  secondaryAction: "Join VisCo",
};

/* ── Quem somos ──────────────────────────────────────────────────────────── */

export const ABOUT = {
  title: "About VisCo",
  // O primeiro parágrafo aparece maior que os demais.
  paragraphs: [
    "Founded in 2025, the Visual Computing Research Group (VisCo) develops computational methods for signal processing, visual computing, and artificial intelligence.",
    "Our research focuses on developing efficient computational methods for representing, processing, and understanding visual information, combining mathematical foundations with modern artificial intelligence.",
    "Based at the Technology Center (CT) of the Federal University of Santa Maria (UFSM), Brazil, VisCo brings together undergraduate and graduate students, faculty members, and collaborators from UFSM and partner institutions to conduct interdisciplinary research and foster scientific collaboration."
  ],
  newsTitle: "Recent",
};

/* ── Linhas de pesquisa ──────────────────────────────────────────────────────
 * O `id` liga publicações e projetos à linha. Ao criar uma linha nova,
 * acrescente o id em LineId abaixo e use o mesmo id nas publicações. */

export type LineId = "nvc" | "cv" | "osa" | "sip" | "vdm";

export const RESEARCH = {
  title: "Research lines",
  intro: "Lines with published work link to the corresponding papers.",
  activeLabel: "Active projects",
  concludedLabel: "Concluded projects",
};

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

/* ── Projetos ────────────────────────────────────────────────────────────────
 * `active: true` aparece em "Active projects"; `false` vai para a lista
 * recolhida de concluídos. `code: null` esconde o link de código. */

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

/* ── Publicações ─────────────────────────────────────────────────────────────
 * A página agrupa sozinha por ano, do mais novo para o mais antigo. Basta
 * acrescentar o item na lista, em qualquer posição.
 * `type` aceita "Journal" ou "Conference".
 * `pdf`, `doi` e `code`: use null para esconder o link. */

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

export const PUBLICATIONS_TEXT = {
  title: "Papers by year",
  searchPlaceholder: "Search title, venue, author",
  olderLabel: "Earlier",
};

/* Publicações com mais de OLDER_AFTER anos saem da listagem principal e vão
   para um bloco recolhido, que começa fechado. Contado a partir do ano atual:
   com 5, em 2026 tudo de 2020 para trás é dobrado. */
export const OLDER_AFTER = 5;

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
  {
    year: 2020,
    title: "Projection formats for omnidirectional video coding",
    authors: "Silveira, T. L. T. et al.",
    venue: "SIBGRAPI 2020",
    type: "Conference",
    line: "nvc",
    pdf: "#",
    doi: "#",
    code: null,
  },
];

/* ── Pessoas ─────────────────────────────────────────────────────────────────
 * `photo`: caminho ou URL da foto. Para usar um arquivo do repositório,
 * coloque a imagem em `public/people/` e escreva "/people/nome.jpg".
 * `link`: para onde o nome aponta (Lattes, GitHub, página pessoal). */

export type Person = {
  name: string;
  role: string;
  area: string;
  photo: string;
  link: string;
};

export const PEOPLE_TEXT = {
  title: "Meet the team",
  intro: "Names link to each person's Lattes or GitHub profile.",
  facultyLabel: "Faculty",
  studentsLabel: "Current students",
  alumniLabel: "Alumni",
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

/* Registro de quem passou pelo grupo. */
export const ALUMNI: { name: string; degree: string }[] = [
  { name: "Former Student A", degree: "M.Sc. 2023" },
  { name: "Former Student B", degree: "B.Sc. 2022" },
];

/* ── Notícias (carrossel do "Who we are") ────────────────────────────────────
 * `tag` aceita: "Publication", "Event", "Project", "Resource", "Opportunity".
 * `href`: use null para um item sem link. Para apontar para uma seção da
 * própria página, use "#research", "#publications", "#join". */

export type NewsTag =
  | "Publication"
  | "Event"
  | "Project"
  | "Resource"
  | "Opportunity";

export type NewsItem = {
  date: string;
  tag: NewsTag;
  text: string;
  img: string;
  href: string | null;
};

export const NEWS: NewsItem[] = [
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

/* ── Onde estamos (textos) ───────────────────────────────────────────────── */

export const LOCATION_TEXT = {
  title: "Where we are",
  note: "To visit, send an email in advance so someone can meet you at the building entrance.",
  caption: "Centro de Tecnologia · Campus Camobi",
};

/* ── Participe ───────────────────────────────────────────────────────────── */

export const JOIN = {
  title: "How to join",
  columns: [
    {
      label: "Students",
      body: [
        "The group supervises M.Sc. and Ph.D. candidates and undergraduate researchers in Computer Science and Electrical Engineering.",
        "A background in image processing, computer vision or machine learning is helpful.",
        "CAPES and CNPq scholarships are available when positions open. Current openings are listed under Recent, in the home section.",
      ],
    },
    {
      label: "Collaborators",
      body: [
        "VisCo co-advises students, publishes jointly, and takes part in funded research with academic groups and industry partners.",
        "The group is affiliated with SBC, SBMAC and IEEE, and registered with PRPGP/UFSM.",
        "Collaborations are open in optics, coding, perception and robotics applied to omnidirectional imaging.",
      ],
    },
  ],
  ctaNote:
    "Write to Prof. Thiago Silveira with a short description of your background and research interests.",
};
