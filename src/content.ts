/* ============================================================================
 * CONTEÚDO DO SITE — edite tudo por aqui
 * ============================================================================
 *
 * Este é o único arquivo que precisa ser alterado para atualizar o site:
 * textos, pessoas, publicações, projetos, notícias e localização. 
 * Nenhum componente precisa ser tocado.  :)
 *
 * Regras para não quebrar nada: 
 *   - Mantenha as aspas e as vírgulas exatamente como estão.
 *   - Aspas dentro de um texto precisam de barra invertida: "o \"cubo\"".
 *   - Campos marcados como `null` significam "não existe"; para preencher,
 *     troque null por um texto entre aspas.
 *   - Os identificadores das linhas de pesquisa (LineId) ligam publicações e
 *     projetos às linhas. Se criar uma linha nova, adicione o id na lista
 *     LineId logo abaixo.
 *   - Quebra de linha dentro de um texto: escreva \n no meio da string.
 *     Ex.: role: "Lead Faculty\nHead of Lab". Funciona nos campos de pessoas.
 *   - Imagens locais: coloque o arquivo em `public/photos/` e escreva o
 *     caminho começando com barra: photo: "/photos/tltsilveira.jpg".
 *     (Pastas fora de `public/` não são servidas pelo site.)
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
    "Technology Center (CT)",
    "Federal University of Santa Maria (UFSM)",
    "Santa Maria, RS, Brazil",
    "Camobi Campus",
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
    "The Visual Computing Research Group (VisCo) develops efficient computational methods for signal processing, visual computing, and artificial intelligence.",
    "Research focuses on representing, processing, compressing, and understanding visual information by combining mathematical foundations with modern artificial intelligence.",
    "Activities emphasize fundamental and applied research, interdisciplinary collaboration, open science, and reproducible computational research.",
    "Founded in 2025, VisCo is based at the Technology Center (CT) of the Federal University of Santa Maria (UFSM), Brazil, bringing together faculty members, students, and collaborators from partner institutions."
  ],
  newsTitle: "News",
};

/* ── Linhas de pesquisa ──────────────────────────────────────────────────────
 * O `id` liga publicações e projetos à linha. Ao criar uma linha nova,
 * acrescente o id em LineId abaixo e use o mesmo id nas publicações. */

export type LineId = "sr" | "sip" | "vu" | "ci" | "gvc";

export const RESEARCH = {
  title: "Research",
  intro: "",
  activeLabel: "Projects (Active)",
  concludedLabel: "Projects (Concluded)",
};

export const LINES: { id: LineId; title: string; desc: string }[] = [
  {
    id: "sr",
    title: "Signal Representation",
    desc: "Efficient representations, transform methods, and approximation techniques for visual and multidimensional signals.",
  },
  {
    id: "sip",
    title: "Signal & Image Processing",
    desc: "Signal, image, and video processing, restoration, enhancement, filtering, and computational imaging.",
  },
  {
    id: "vu",
    title: "Visual Understanding",
    desc: "Computer vision, pattern recognition, scene understanding, and semantic analysis.",
  },
  {
    id: "ci",
    title: "Computational Intelligence",
    desc: "Machine learning, deep learning, optimization, and intelligent data analysis.",
  },
  {
    id: "gvc",
    title: "Geometric Visual Computing",
    desc: "Geometry-aware methods for omnidirectional imaging, immersive media, and spatial visual data.",
  },
];

/* ── Projetos ────────────────────────────────────────────────────────────────
 * `active: true` aparece em "Active projects"; `false` vai para a lista
 * recolhida de concluídos. `code: null` esconde o link de código.
 * `lines` aceita uma ou mais áreas: lines: ["gvc"] ou lines: ["gvc", "ci"]. */

export type Project = {
  title: string;
  year: string;
  desc: string;
  lines: LineId[];
  tags: string[];
  active: boolean;
  code: string | null;
};

export const PROJECTS: Project[] = [
  {
    title: "Low-Complexity Methods for End-to-End Neural Compression of 360° Images and Videos",
    year: "2024–27",
    desc: "This project advances the state of the art in low-complexity neural compression of 360° images and videos through efficient deep learning architectures, spherical data representations, and model optimization techniques.",
    lines: ["gvc", "ci"],
    tags: ["T. L. T. Silveira", "FAPERGS PqG"],
    active: true,
    code: null,
  },
];

/* ── Publicações ─────────────────────────────────────────────────────────────
 * A página agrupa sozinha por ano, do mais novo para o mais antigo. Basta
 * acrescentar o item na lista, em qualquer posição.
 * `type` aceita "Journal" ou "Conference" — aparece como a bolinha colorida
 *   antes do título (roxa = journal, verde = conference).
 * `lines` aceita uma ou mais áreas: lines: ["sr"] ou lines: ["sr", "gvc"].
 * `collab: true` marca papers com colaboradores de fora do VisCo — mostra a
 *   etiqueta "Collaboration" na entrada. Omita o campo nos demais.
 * `pdf`, `doi` e `code`: use null para esconder o link. */

export type PubType = "Journal" | "Conference";

export type Pub = {
  year: number;
  title: string;
  authors: string;
  venue: string;
  type: PubType;
  lines: LineId[];
  collab?: boolean;
  pdf: string | null;
  doi: string | null;
  code: string | null;
};

export const PUBLICATIONS_TEXT = {
  title: "Publications",
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
    title: "Low-Complexity Compression of 360° Still Images",
    authors: "Bastos, B. M., Segala, E. B., and Silveira, T. L. T.",
    venue: "IEEE Latin American Symposium on Circuits and Systems",
    type: "Conference",
    lines: ["sr", "gvc"],
    pdf: null,
    doi: "https://doi.org/10.1109/LASCAS64004.2025.10966281",
    code: null,
  },
  {
    year: 2025,
    title: "Superpixel-driven 360° Image Compression",
    authors: "Binkowski, B., Segala, E. B., and Silveira, T. L. T.",
    venue: "Springer Multimedia Tools and Applications",
    type: "Journal",
    lines: ["gvc", "sr"],
    pdf: null,
    doi: "https://doi.org/10.1007/s11042-025-20876-1",
    code: null,
  },
  {
    year: 2025,
    title: "Complexity-Reduced End-to-End Fetal ECG Signal Recovery and QRS Complex Detection",
    authors: "Remus, J. C. and Silveira, T. L. T.",
    venue: "Simpósio Brasileiro de Computação Aplicada à Saúde",
    type: "Conference",
    lines: ["sip"],
    pdf: null,
    doi: "https://doi.org/10.5753/sbcas.2025.7731",
    code: null,
  },
  {
    year: 2025,
    title: "Anchor-based Gravity Alignment for Panoramas",
    authors: "Bergmann, M. A., Stringhini, R. M., Silveira, T. L. T., and Jung, C. R.",
    venue: "IEEE International Conference on Image Processing",
    type: "Conference",
    lines: ["gvc", "vu"],
    collab: true,
    pdf: null,
    doi: "https://doi.org/10.1109/ICIP55913.2025.11084672",
    code: null,
  },
  {
    year: 2026,
    title: "Evaluating Contactless Fingerprint Segmentation for Interoperable Biometric Identification Systems",
    authors: "Arcoverde Neto, E. N. and Silveira, T. L. T.",
    venue: "Revista de Informática Teórica e Aplicada",
    type: "Journal",
    lines: ["vu"],
    pdf: "https://seer.ufrgs.br/index.php/rita/article/view/147966/98412",
    doi: "https://doi.org/10.22456/2175-2745.147966",
    code: null,
  },
  {
    year: 2026,
    title: "Do Deep Learning Models Generalize Facial Emotion Recognition in Different Age Groups?",
    authors: "Evangelista, N. L. and Silveira, T. L. T.",
    venue: "Revista de Informática Teórica e Aplicada",
    type: "Journal",
    lines: ["vu"],
    pdf: "https://seer.ufrgs.br/index.php/rita/article/view/144479/98280",
    doi: "https://doi.org/10.22456/2175-2745.144479",
    code: null,
  },
  {
    year: 2026,
    title: "Exploring Asymmetric Autoencoder Architectures for Computationally-Efficient Neural Image Compression",
    authors: "Augusto, L. S., Arguilar, V. A., Silveira, T. L. T., and Grellert, M.",
    venue: "IEEE Design & Test",
    type: "Journal",
    lines: ["ci", "sr"],
    collab: true,
    pdf: null,
    doi: "https://doi.org/10.1109/MDAT.2025.3615794",
    code: null,
  },
  {
    year: 2026,
    title: "Adapting Convolutions for Effective Omnidirectional Image Processing",
    authors: "Stringhini, R. M., Silveira, T. L. T., and Jung, C. R.",
    venue: "Journal of the Brazilian Computer Society",
    type: "Journal",
    lines: ["ci", "gvc"],
    collab: true,
    pdf: "https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5654/4048",
    doi: "https://doi.org/10.5753/jbcs.2026.5654",
    code: null,
  },
];

/* ── Pessoass ─────────────────────────────────────────────────────────────────
 * `photo`: caminho ou URL da foto. Para foto local, coloque o arquivo em
 * `public/photos/` e escreva "/photos/nome.jpg" (com a barra inicial).
 * `photo: ""` mostra um círculo neutro; `link: ""` deixa o nome sem link.
 * Para quebrar linha em `role` ou `area`, use \n dentro do texto. */

export type Person = {
  name: string;
  role: string;
  area: string;
  photo: string;
  link: string;
};

export const PEOPLE_TEXT = {
  title: "People",
  intro: "",
  facultyLabel: "Faculty",
  studentsLabel: "Students",
  alumniLabel: "Alumni",
};

export const FACULTY: Person[] = [
  {
    name: "Thiago L. T. da Silveira",
    role: "Lead Faculty",
    area: "signal/image processing · pattern recognition · computer vision",
    photo: "/photos/tltsilveira.jpg",
    link: "https://tltsilveira.github.io/",
  },
  {
    name: "Adriano Q. de Oliveira",
    role: "Affiliated Faculty",
    area: "applied AI · image processing · computer vision",
    photo: "/photos/aqoliveira.png",
    link: "https://adrianoquiliao.github.io/",
  },
  {
    name: "Leonardo R. Emmendorfer",
    role: "Affiliated Faculty",
    area: "",
    photo: "/photos/lremmendorfer.png",
    link: "http://lattes.cnpq.br/1129100746134234",
  },
  {
    name: "Luiz J. S. Silva",
    role: "Affiliated Faculty",
    area: "",
    photo: "",
    link: "#",
  },
];

export const STUDENTS: Person[] = [
  {
    name: "Alejandro V. T. Brites",
    role: "Thesis Student",
    area: "Volume segmentation",
    photo: "",
    link: "",
  },
  {
    name: "Alice Z. Marques",
    role: "Thesis Student",
    area: "Signal processing",
    photo: "",
    link: "",
  },
  {
    name: "Artur C. Segat",
    role: "UG Researcher",
    area: "Neural compression",
    photo: "",
    link: "",
  },
  {
    name: "Deomar S. S. Junior",
    role: "Grad Researcher",
    area: "Neural compression",
    photo: "/photos/dssjunior.jpg",
    link: "https://www.linkedin.com/in/deomar/",
  },
  {
    name: "Diego R. Chaves",
    role: "UG Researcher | Thesis Student",
    area: "Neural compression",
    photo: "/photos/drchaves.jpg",
    link: "https://github.com/drchaves",
  },
  {
    name: "Eduardo M. da Silveira",
    role: "MSc Student",
    area: "Gaussian splatting",
    photo: "/photos/emsilveira.jpeg",
    link: "https://github.com/eduardomsilveira",
  },
  {
    name: "Enzo B. Segala",
    role: "UG Researcher",
    area: "Transform coding",
    photo: "/photos/ebsegala.jpg",
    link: "https://lattes.cnpq.br/5030432474101116",
  },
  {
    name: "Felipe A. Andrade",
    role: "UG Researcher",
    area: "Object detection",
    photo: "",
    link: "",
  },
  {
    name: "Gabriel S. Baggio",
    role: "UG Researcher | Thesis Student",
    area: "Neural compression",
    photo: "/photos/gsbaggio.png",
    link: "https://github.com/gsbaggio",
  },
  {
    name: "Giovanni Gaiardo",
    role: "MSc Student",
    area: "Neural compression",
    photo: "/photos/ggaiardo.jpg",
    link: "",
  },
  {
    name: "Henrique P. Gerhardt",
    role: "MSc Student",
    area: "Optical flow",
    photo: "",
    link: "",
  },
  {
    name: "Jonathan Cardozo",
    role: "Thesis Student",
    area: "Transform coding",
    photo: "/photos/jcardozo.jpeg",
    link: "https://github.com/Joncardozo",
  },
  {
    name: "Lucas B. S. Ribeiro",
    role: "Thesis Student",
    area: "Neural compression",
    photo: "/photos/lribeiro.jpeg",
    link: "https://github.com/lucasbsribeiro",
  },
  {
    name: "Luis G. W. Tozevich",
    role: "UG Researcher",
    area: "Object detection",
    photo: "",
    link: "",
  },
  {
    name: "Luiza E. Silveira",
    role: "UG Researcher",
    area: "Neural compression",
    photo: "/photos/lespinoza.jpg",
    link: "http://lattes.cnpq.br/0346792057486966",
  },
  {
    name: "Maria C. B. Silva",
    role: "Thesis Student",
    area: "Signal processing",
    photo: "/photos/mcsilva.png",
    link: "",
  },
  {
    name: "Maria R. Piekas",
    role: "UG Researcher",
    area: "Signal processing",
    photo: "",
    link: "",
  },
  {
    name: "Yasmin F. Garcia",
    role: "UG Researcher",
    area: "Neural compression",
    photo: "/photos/ygarcia.jpeg",
    link: "https://www.linkedin.com/in/yasmin-garcia-63563428b/",
  },
];

/* Registro de quem passou pelo grupo. */
export const ALUMNI: { name: string; degree: string }[] = [
  { name: "Bruno Binkowski", degree: "M.Sc. 2025" },
  { name: "Julia C. Remus", degree: "M.Sc. 2025" },
  { name: "Manuel S. T. Veras", degree: "M.Sc. 2025" },
  { name: "Bruno M. Bastos", degree: "B.Sc. 2025" },
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
  /*{
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
  },*/
  {
    date: "Aug 2026",
    tag: "Publication",
    text: "Some paper has been accepted for publication in ...",
    img: "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=800&h=500&fit=crop&auto=format",
    href: "#publications",
  },
];

/* ── Onde estamos (textos) ───────────────────────────────────────────────── */

export const LOCATION_TEXT = {
  title: "Location",
  note: "Visitors are kindly asked to email us in advance to arrange access to the building.",
  caption: "CT · UFSM",
};

/* ── Participe ───────────────────────────────────────────────────────────── */

export const JOIN = {
  title: "Join VisCo",
  columns: [
    {
      label: "Students",
      body: [
        "VisCo welcomes undergraduate and graduate students interested in signal processing, visual computing, and artificial intelligence.",
        "Students participate in cutting-edge research, scientific publications, and collaborations with national and international partners.",
        "Interested students are encouraged to contact us to discuss research opportunities.",
      ],
    },
    {
      label: "Collaborators",
      body: [
        "VisCo welcomes collaborations with researchers, universities, research centers, and industry partners.",
        "Current collaborations include joint research projects, co-supervision of students, scientific publications, and funded research initiatives.",
        "Researchers interested in collaborative projects are encouraged to get in touch.",
      ],
    },
  ],
  ctaNote:
    "Interested in joining or collaborating with VisCo? We'd be happy to hear from you.",
};
