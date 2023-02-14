import { fileURLToPath, URL } from "url";

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import Components from "unplugin-vue-components/vite";
import vuetify from "@vuetify/vite-plugin";

// https://vitejs.dev/config/
export default defineConfig({
  root: __dirname,
  publicDir: "static",
  plugins: [
    vue(),
    vuetify({
      autoImport: true,
    }),
    Components({
      dirs: ["views", "components"],
      directoryAsNamespace: true,
    }),
  ],
  define: { "process.env": {} },

  resolve: {
    alias: {
      "@": fileURLToPath(new URL(".", import.meta.url)),
    },
  },
  build: {
    outDir: "../public",
    emptyOutDir: true,
  },
});
