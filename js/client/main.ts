import { createApp } from "vue";
import { pinia } from "./stores/store.pinia";
import router from "./router";
import App from "./App.vue";

const app = createApp(App);

app.use(pinia);
app.use(router);

app.mount("#app");
