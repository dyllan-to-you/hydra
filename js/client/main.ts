import { createApp } from "vue";
import { pinia } from "./stores/store.pinia";
import router from "./router";
import App from "./App.vue";
import vuetify from "./plugins/vuetify";
import { loadFonts } from "./plugins/webfontloader";

import { Vue3ProgressPlugin } from "@marcoschulte/vue3-progress";
import UplotVue from "uplot-vue";

loadFonts();

const app = createApp(App);

app.use(vuetify);
app.use(pinia);
app.use(router);

app.use(Vue3ProgressPlugin);
app.component("u-plot", UplotVue);

app.mount("#app");
