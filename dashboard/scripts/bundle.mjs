/**
 * Bundles the built dashboard (dist/) into the single-file HTML template
 * shipped with the TORAX Python package: torax/_src/plotting/dashboard.html.
 *
 * The template is fully self-contained (inline JS/CSS, Roboto embedded from
 * dashboard/fonts/) and carries a `TORAX_RUNS` placeholder that
 * torax/_src/plotting/dashboard.py replaces with exported run documents.
 *
 * Usage: npm run bundle   (runs `vite build` first, then this script)
 */
import {readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const dashboardDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const distAssets = join(dashboardDir, 'dist', 'assets');
const outputPath = join(
  dashboardDir,
  '..',
  'torax',
  '_src',
  'plotting',
  'dashboard.html',
);

/** Marker replaced by dashboard.py with a JSON array of run documents. */
const RUNS_PLACEHOLDER = '/*__TORAX_RUNS__*/ []';

const files = readdirSync(distAssets);
const asset = suffix => {
  const name = files.find(f => f.endsWith(suffix));
  if (!name) throw new Error(`No ${suffix} asset in dist/ — run vite build`);
  return readFileSync(join(distAssets, name), 'utf8');
};
// The </script guard keeps the inline script from being terminated early.
const js = asset('.js').replace(/<\/script/gi, '<\\/script');
const css = asset('.css');

const fontFace = weight => {
  const data = readFileSync(
    join(dashboardDir, 'fonts', `roboto-${weight}.woff2`),
  ).toString('base64');
  return `@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: ${weight};
  font-display: swap;
  src: url(data:font/woff2;base64,${data}) format('woff2');
}`;
};

const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="color-scheme" content="light dark" />
    <title>TORAX Dashboard</title>
    <style>
${[400, 500, 700].map(fontFace).join('\n')}
${css}
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script>
      window.TORAX_EMBEDDED_RUNS = ${RUNS_PLACEHOLDER};
    </script>
    <script type="module">
${js}
    </script>
  </body>
</html>
`;

writeFileSync(outputPath, html);
console.log(
  `Wrote ${outputPath} (${(html.length / 1024 / 1024).toFixed(2)} MB)`,
);
