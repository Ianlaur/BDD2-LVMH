/// <reference types="vite/client" />

// CSS module declarations
declare module '*.css' {
  const content: string;
  export default content;
}

// Extend CSSProperties to support CSS custom properties
import 'react';
declare module 'react' {
  interface CSSProperties {
    [key: `--${string}`]: string | number | undefined;
  }
}
