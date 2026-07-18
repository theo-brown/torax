/** Minimal LaTeX-subset parser for physics labels.
 *
 * Supports the notation used by the TORAX plot configs: `text $math$ text`
 * segments, and inside math mode `_{sub}` / `^{sup}` (braced or single
 * character), `\mathrm{...}` upright groups, Greek-letter commands,
 * `\hat/\dot/\bar/\tilde` accents, `\langle`/`\rangle`, `~` spaces.
 * Letters in math mode are italic (LaTeX convention); digits and
 * punctuation stay upright.
 *
 * Two renderers consume the parse: VarLabel.tsx (React DOM) and
 * labelToPlotlyHtml below (Plotly's pseudo-HTML: <i>, <sub>, <sup>).
 */

export type TexNode =
  | {kind: 'text'; value: string; italic: boolean}
  | {kind: 'sub' | 'sup'; children: TexNode[]};

export type TexSegment =
  {math: true; nodes: TexNode[]} | {math: false; text: string};

/** Lowercase Greek renders italic (LaTeX convention: it names a variable);
 *  `\mathrm{...}` overrides this for anything inside it. */
// prettier-ignore
const ITALIC_SYMBOLS: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', zeta: 'ζ',
  eta: 'η', theta: 'θ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ',
  nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ',
  upsilon: 'υ', phi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
};

/** Uppercase Greek, delimiters, and operators render upright. */
// prettier-ignore
const UPRIGHT_SYMBOLS: Record<string, string> = {
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π',
  Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω', langle: '⟨', rangle: '⟩',
  infty: '∞', cdot: '·', times: '×', propto: '∝', pm: '±', partial: '∂',
  nabla: '∇',
};

/** Combining accent characters appended to the accented glyph. */
const ACCENTS: Record<string, string> = {
  hat: '̂',
  dot: '̇',
  bar: '̄',
  tilde: '̃',
};

/** Returns [group content, index after group]. `{...}` or a single char. */
function grabGroup(s: string, i: number): [string, number] {
  if (s[i] !== '{') return [s[i] ?? '', i + 1];
  let depth = 1;
  let j = i + 1;
  while (j < s.length && depth > 0) {
    if (s[j] === '{') depth++;
    else if (s[j] === '}') depth--;
    j++;
  }
  return [s.slice(i + 1, depth === 0 ? j - 1 : j), j];
}

/** Returns [subscript/superscript source, index after it]. Handles a braced
 *  group, a command with an optional braced argument (`_\mathrm{i}`), or a
 *  single character. */
function grabToken(s: string, i: number): [string, number] {
  if (s[i] === '{') return grabGroup(s, i);
  if (s[i] === '\\') {
    const match = /^\\[a-zA-Z]+/.exec(s.slice(i));
    if (match) {
      const j = i + match[0].length;
      if (s[j] === '{') {
        const [, next] = grabGroup(s, j);
        return [s.slice(i, next), next];
      }
      return [match[0], j];
    }
  }
  return [s[i] ?? '', i + 1];
}

function mergeText(nodes: TexNode[]): TexNode[] {
  const out: TexNode[] = [];
  for (const node of nodes) {
    const prev = out[out.length - 1];
    if (
      node.kind === 'text' &&
      prev?.kind === 'text' &&
      prev.italic === node.italic
    ) {
      prev.value += node.value;
    } else {
      out.push(node);
    }
  }
  return out;
}

function parseMath(s: string, upright: boolean): TexNode[] {
  const nodes: TexNode[] = [];
  const text = (value: string, italic: boolean) =>
    nodes.push({kind: 'text', value, italic});
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === '\\') {
      const match = /^\\([a-zA-Z]+)/.exec(s.slice(i));
      if (match) {
        const cmd = match[1];
        i += match[0].length;
        if (cmd === 'mathrm' || cmd === 'text' || cmd === 'rm') {
          const [content, next] = grabGroup(s, i);
          nodes.push(...parseMath(content, true));
          i = next;
        } else if (cmd in ACCENTS) {
          const [content, next] = grabGroup(s, i);
          const inner = parseMath(content, upright);
          const last = inner[inner.length - 1];
          if (last?.kind === 'text') last.value += ACCENTS[cmd];
          nodes.push(...inner);
          i = next;
        } else if (cmd in ITALIC_SYMBOLS || cmd in UPRIGHT_SYMBOLS) {
          const glyph = ITALIC_SYMBOLS[cmd] ?? UPRIGHT_SYMBOLS[cmd];
          text(glyph, cmd in ITALIC_SYMBOLS && !upright);
        } else {
          text(cmd, false); // Unknown command: show its name.
        }
        continue;
      }
      if (s[i + 1] === ',') {
        text(' ', false); // \, thin space
        i += 2;
        continue;
      }
      i++;
      continue;
    }
    if (ch === '_' || ch === '^') {
      const [content, next] = grabToken(s, i + 1);
      nodes.push({
        kind: ch === '_' ? 'sub' : 'sup',
        children: parseMath(content, upright),
      });
      i = next;
      continue;
    }
    if (ch === '{') {
      const [content, next] = grabGroup(s, i);
      nodes.push(...parseMath(content, upright));
      i = next;
      continue;
    }
    if (ch === '~') {
      text(' ', false);
      i++;
      continue;
    }
    if (/[a-zA-Z]/.test(ch)) {
      text(ch, !upright);
      i++;
      continue;
    }
    text(ch, false); // digits, punctuation, spaces
    i++;
  }
  return mergeText(nodes);
}

/** Splits `text $math$ text` into segments with math segments parsed. */
export function parseLabel(label: string): TexSegment[] {
  const segments: TexSegment[] = [];
  const re = /\$([^$]*)\$/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(label)) !== null) {
    if (match.index > last) {
      segments.push({math: false, text: label.slice(last, match.index)});
    }
    segments.push({math: true, nodes: parseMath(match[1], false)});
    last = match.index + match[0].length;
  }
  if (last < label.length) {
    segments.push({math: false, text: label.slice(last)});
  }
  return segments;
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function nodesToHtml(nodes: TexNode[]): string {
  let html = '';
  for (const node of nodes) {
    if (node.kind === 'text') {
      const value = escapeHtml(node.value);
      html += node.italic ? `<i>${value}</i>` : value;
    } else {
      const tag = node.kind;
      html += `<${tag}>${nodesToHtml(node.children)}</${tag}>`;
    }
  }
  return html;
}

/** Renders a label to the pseudo-HTML subset Plotly understands. */
export function labelToPlotlyHtml(label: string): string {
  return parseLabel(label)
    .map(seg => (seg.math ? nodesToHtml(seg.nodes) : escapeHtml(seg.text)))
    .join('');
}
