import { createApp } from "vue";
import { pinia } from "./stores/store.pinia";
import router from "./router";
import App from "./App.vue";

import { Vue3ProgressPlugin } from "@marcoschulte/vue3-progress";

const app = createApp(App);

app.use(pinia);
app.use(router);

app.use(Vue3ProgressPlugin);

app.mount("#app");
