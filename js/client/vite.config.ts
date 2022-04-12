import { fileURLToPath, URL } from "url";

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import Components from "unplugin-vue-components/vite";

// https://vitejs.dev/config/
export default defineConfig({
  root: __dirname,
  publicDir: "static",
  plugins: [
    vue(),
    Components({
      dirs: ["views", "components"],
      directoryAsNamespace: true,
      resolvers: [
        (name) => {
          switch (name) {
            case "Uplot":
              return { importName: "default", path: "uplot-vue" };
            default:
              break;
          }
        },
      ],
    }),
  ],
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
