declare module 'plotly.js-basic-dist-min' {
  const Plotly: {
    react: (
      root: HTMLElement,
      data: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ) => Promise<unknown>;
    purge: (root: HTMLElement) => void;
  };
  export default Plotly;
}
