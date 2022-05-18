import { createApp } from "vue";
import { pinia } from "./stores/store.pinia";
import router from "./router";
import App from "./App.vue";

import { Vue3ProgressPlugin } from "@marcoschulte/vue3-progress";
import UplotVue from "uplot-vue";
import Slider from "@vueform/slider";
import "@vueform/slider/themes/default.css";

const app = createApp(App);

app.use(pinia);
app.use(router);

app.use(Vue3ProgressPlugin);
app.component("vueform-slider", Slider);
app.component("u-plot", UplotVue);

app.mount("#app");
