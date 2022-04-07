<script lang="ts">
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

const clamp = (num, min, max) => Math.min(Math.max(num, min), max);

/*
  // https://newbedev.com/javascript-math-random-normal-distribution-gaussian-bell-curve
  function randn_bm(min, max, skew) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );

    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) num = randn_bm(min, max, skew); // resample between 0 and 1 if out of range
    num = Math.pow(num, skew); // Skew
    num *= max - min; // Stretch to fill range
    num += min; // offset to min
    return num;
  }
*/

// adapted from http://jsfiddle.net/Xotic750/3rfT6/
const boxMullerRandom = (function () {
  let phase = 0,
    random = Math.random,
    x1,
    x2,
    w,
    z;

  return () => {
    if (!phase) {
      do {
        x1 = 2.0 * random() - 1.0;
        x2 = 2.0 * random() - 1.0;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1.0);

      w = Math.sqrt((-2.0 * Math.log(w)) / w);
      z = x1 * w;
    } else {
      z = x2 * w;
    }

    phase ^= 1;

    return z;
  };
})();

function randomWalk(steps, value = 0, min = -100, max = 100) {
  steps = steps >>> 0 || 100;
  let randFunc = boxMullerRandom;

  let points = [],
    t;

  for (t = 0; t < steps; t += 1) {
    let extra = randFunc();
    let newVal = value + extra;

    if (newVal > max || newVal < min) value = clamp(value - extra, min, max);
    else value = newVal;

    points.push(value);
  }

  return points;
}

function addRandData(data, howMany, min, max) {
  let last = data[length - 1];
  return data.slice(1).concat(randomWalk(howMany, last, min, max));
}

const length = 600;

export default {
  name: "RandomWalk",
  // components: { UplotVue },
  data() {
    return {
      options: {
        title: "6 series x 600 points @ 60fps",
        width: document.body.clientWidth,
        height: 600,
        pxAlign: false,
        scales: {
          y: {
            //	auto: false,
            range: [-6, 6],
          },
        },
        axes: [
          {
            space: 300,
          },
        ],
        series: [
          {},
          {
            label: "Sine",
            stroke: "red",
            fill: "rgba(255,0,0,0.1)",
          },
          {
            stroke: "green",
            fill: "#4caf505e",
          },
          {
            stroke: "blue",
            fill: "#0000ff20",
          },
          {
            stroke: "orange",
            fill: "#ffa5004f",
          },
          {
            stroke: "magenta",
            fill: "#ff00ff20",
          },
          {
            stroke: "purple",
            fill: "#80008020",
          },
        ],
      },
      shift: length - 1,
    };
  },
  methods: {
    now() {
      const now = Math.floor(Date.now() / 1e3);
      return now;
    },
    onCreate(chart: uPlot) {
      console.log("Created from render fn");
    },
    onDelete(chart: uPlot) {
      console.log("Deleted from render fn");
    },
    toRadians(angle) {
      return angle * (Math.PI / 180);
    },
    update() {
      this.shift += 1;
      let [xs, sin, _1, _2, _3, _4, _5] = this.dataset;

      const data = [
        [...xs.slice(1), this.now() + this.shift * 60 * 5],
        [...sin.slice(1), Math.sin(this.shift / 16) * 5],
        [...addRandData(_1, 1, -6, -1)],
        [...addRandData(_2, 1, -6, -1)],
        [...addRandData(_3, 1, -2, 2)],
        [...addRandData(_4, 1, -1, 6)],
        [...addRandData(_5, 1, -1, 6)],
      ];
      this.dataset = data;
      console.log(this.dataset);
      this.$forceUpdate();
      requestAnimationFrame(this.update);
    },
  },
  beforeMount() {
    // Initialize data inside mounted hook, to prevent Vue from adding watchers, otherwise performance becomes unbearable
    let data = [
      Array.from({ length: length }, (v, i) => this.now() + i * 60 * 5),
      Array.from({ length: length }, (v, i) => Math.sin(i / 16) * 5),
      randomWalk(length, -4, -6, 1),
      randomWalk(length, -2, -6, 1),
      randomWalk(length, 0, -2, 2),
      randomWalk(length, 2, -1, 6),
      randomWalk(length, 4, -1, 6),
    ];

    console.log(data);
    this.dataset = data;
  },
  mounted() {
    // setInterval(() => {
    //   this.update();
    // }, 1000);
    this.update();
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
