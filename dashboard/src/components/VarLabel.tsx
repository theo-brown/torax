import type {ReactNode} from 'react';
import {parseLabel, type TexNode} from '../latex';

function renderNodes(nodes: TexNode[]): ReactNode[] {
  return nodes.map((node, i) => {
    if (node.kind === 'text') {
      return node.italic ? <i key={i}>{node.value}</i> : node.value;
    }
    return node.kind === 'sub' ? (
      <sub key={i}>{renderNodes(node.children)}</sub>
    ) : (
      <sup key={i}>{renderNodes(node.children)}</sup>
    );
  });
}

/** Renders a label with `text $math$` LaTeX-subset markup (see latex.ts).
 *  Math segments are set in an italic serif math face; everything is
 *  emitted as React text nodes, so arbitrary variable names from data
 *  files are safe to display. */
export function VarLabel({label}: {label: string}) {
  return (
    <>
      {parseLabel(label).map((seg, i) =>
        seg.math ? (
          <span className="tex" key={i}>
            {renderNodes(seg.nodes)}
          </span>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </>
  );
}
