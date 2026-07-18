import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

// base './' so the built app can be served from any sub-path.
export default defineConfig({
  base: './',
  plugins: [react()],
});
