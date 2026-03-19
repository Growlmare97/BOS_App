/// <reference types="vite/client" />

declare module 'plotly.js-dist-min' {
  export * from 'plotly.js';
  export { default } from 'plotly.js';
}

declare module 'react-plotly.js/factory' {
  import { ComponentClass } from 'react';
  import Plotly from 'plotly.js';
  import { Props } from 'react-plotly.js';
  function createPlotlyComponent(plotly: typeof Plotly): ComponentClass<Props>;
  export default createPlotlyComponent;
}
