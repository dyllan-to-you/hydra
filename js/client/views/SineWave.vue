<script lang="ts">
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export default {
  name: "SineWave",
  // components: { UplotVue },
  data() {
    return {
      options: {
        title: "Sine Wave",
        width: document.body.clientWidth,
        height: 600,
        series: [
          {
            label: "Date",
          },
          {
            label: "",
            points: { show: false },
            stroke: "blue",
            fill: "blue",
          },
        ],
        plugins: [],
      },
      counter: 0,
      startTime: Math.floor(Date.now() / 1e3),
    };
  },
  methods: {
    onCreate(chart: uPlot) {
      console.log("Created from render fn");
    },
    onDelete(chart: uPlot) {
      console.log("Deleted from render fn");
    },
    toRadians(angle) {
      return angle * (Math.PI / 180);
    },
  },
  beforeMount() {
    // Initialize data inside mounted hook, to prevent Vue from adding watchers, otherwise performance becomes unbearable
    this.dataset = [
      [...new Array(360)].map((_, i) => this.startTime + i * 60),
      [...new Array(360)].map((_, i) => Math.sin(this.toRadians(i % 360))),
    ];
    this.counter = this.dataset[0].length;
  },
  mounted() {
    setInterval(() => {
      this.counter += 1;
      const data: uPlot.AlignedData = [
        [...this.dataset[0], this.startTime + this.counter * 60],
        [...this.dataset[1], Math.sin(this.toRadians(this.counter % 360))],
      ];
      if (this.counter > 360 * 1.5) {
        data[0].shift();
        data[1].shift();
      }
      this.dataset = data;
      // Since we disabled reactivity for data above
      this.$forceUpdate();
    }, 100);
  },
};
</script>

<template>
  <Uplot
    key="render-key"
    :data="dataset"
    :options="options"
    @delete="onDelete"
    @create="onCreate"
  />
</template>
