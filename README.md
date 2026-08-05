# Site do VisCo · Visual Computing Research Group

Site do grupo em **https://visco-ufsm.github.io/**, feito com [Vite](https://vite.dev) + React.
Este guia cobre como rodar localmente, como editar o conteúdo e como o site é publicado.

## Rodando localmente

### Pré-requisitos

- **Node.js 20 ou superior** — instale a versão LTS em <https://nodejs.org>. Para conferir:

  ```bash
  node -v
  ```

- **pnpm** — já vem embutido no Node, basta habilitar:

  ```bash
  corepack enable
  ```

  Isso usa automaticamente a versão exata do pnpm fixada no `package.json`,
  então não há risco de divergência entre máquinas.

### Passo a passo

```bash
# 1. Baixar o projeto (ou Code → Download ZIP na página do GitHub)
git clone https://github.com/visco-ufsm/visco-ufsm.github.io.git
cd visco-ufsm.github.io

# 2. Instalar as dependências (só na primeira vez)
pnpm install

# 3. Subir o site
pnpm dev
```

Abra **http://localhost:5173** no navegador. Para parar o servidor, `Ctrl+C`.

Não precisa de chave de API nem de configuração extra. A internet é usada na
instalação e, com o site aberto, para as fontes, o mapa do Google e as fotos
de exemplo.

## Editando o conteúdo

**Todo o conteúdo do site vive em um único arquivo: [`src/content.ts`](src/content.ts).**
Textos, pessoas, publicações, projetos, notícias e a localização do mapa —
nenhum outro arquivo precisa ser tocado para atualizações do dia a dia.

Com o `pnpm dev` rodando, salve o arquivo e a página recarrega sozinha.

O arquivo é dividido em blocos comentados. Os mais usados:

| Bloco | O que controla |
|---|---|
| `GROUP` | Nome, e-mail e links do grupo |
| `PLACE` | Endereço e coordenadas — o mapa e o link do Google Maps derivam de `lat`/`lon` |
| `LINES` | As linhas de pesquisa |
| `PROJECTS` | Projetos ativos e concluídos (`active: true/false`) |
| `PUBS` | Publicações — a página agrupa por ano sozinha; basta acrescentar o item |
| `FACULTY` / `STUDENTS` / `ALUMNI` | As pessoas da seção *Meet the team* |
| `NEWS` | Os itens do carrossel *Recent* |
| `OLDER_AFTER` | Publicações com mais de N anos vão para o bloco recolhido *Earlier* |

Regras para não quebrar nada:

- Mantenha aspas e vírgulas exatamente como estão nos itens existentes.
- `null` significa "não existe" (ex.: `code: null` esconde o link de código).
- Para fotos locais, coloque a imagem em `public/people/` e use `"/people/nome.jpg"`.

## Publicando

O deploy é automático: **todo push na branch `main` publica o site** via GitHub
Actions ([.github/workflows/deploy.yml](.github/workflows/deploy.yml)), em cerca
de um minuto. Não é preciso buildar nada manualmente.

```bash
git add -A
git commit -m "atualiza publicações"
git push
```

Para conferir o build de produção localmente antes de subir:

```bash
pnpm build      # gera a pasta dist/
```

## Estrutura do projeto

```
src/
  content.ts            ← conteúdo do site (edite aqui)
  app/
    App.tsx             raiz: navegação e ordem das seções
    sections/           uma seção por arquivo (Hero, About, Research, …)
    components/         componentes compartilhados
      PointCloud.tsx    a animação do topo (superfície de amostras, canvas 2D)
  styles/theme.css      tokens de cor, tipografia e espaçamento
public/
  distortion-analysis/  scripts WS-PSNR/SSIM/MSE (URLs públicas preservadas)
  resources/logo.svg    logo vetorial
```

## Contato

Dúvidas sobre o site: abra uma issue neste repositório ou fale com o grupo em
thiago.silveira@ufsm.br.
