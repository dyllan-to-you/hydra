<script setup lang="ts">
const links = [
  { dest: "Sine", name: "SineWave" },
  { dest: "Random", name: "RandomWalk" },
  { dest: "Prices", name: "PricePrediction" },
];
</script>

<template>
  <header>
    <template v-for="(link, idx) in links" :key="idx">
      <span>
        <router-link :to="{ name: link.dest }">{{ link.name }}</router-link>
      </span>
      <span v-if="idx < links.length - 1"> | </span>
    </template>
  </header>

  <main>
    <router-view></router-view>
  </main>
</template>

<!-- <style>
@import "./assets/base.css";

#app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;

  font-weight: normal;
}

header {
  line-height: 1.5;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

a,
.green {
  text-decoration: none;
  color: hsla(160, 100%, 37%, 1);
  transition: 0.4s;
}

@media (hover: hover) {
  a:hover {
    background-color: hsla(160, 100%, 37%, 0.2);
  }
}

@media (min-width: 1024px) {
  body {
    display: flex;
    place-items: center;
  }

  #app {
    display: grid;
    grid-template-columns: 1fr 1fr;
    padding: 0 2rem;
  }

  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  .logo {
    margin: 0 2rem 0 0;
  }
}
</style> -->
