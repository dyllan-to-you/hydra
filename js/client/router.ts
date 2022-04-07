import * as VueRouter from "vue-router";

export default VueRouter.createRouter({
  history: VueRouter.createWebHistory(),
  routes: [
    {
      path: "/fft-indicator",
      name: "halp",
      component: () => import("./views/halp.vue"),
    },
    {
      path: "/sine",
      name: "Sine",
      component: () => import("./views/SineWave.vue"),
    },
    {
      path: "/random",
      name: "Random",
      component: () => import("./views/RandomWalk.vue"),
    },
    {
      path: "/",
      name: "Prices",
      component: () => import("./views/PricePredictions.vue"),
    },
  ],
});
