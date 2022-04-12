<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from "vue";
import { useProgress } from "@marcoschulte/vue3-progress";

import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import "@marcoschulte/vue3-progress/dist/index.css";

import { api } from "@/feathers";

import {
  columnHighlightPlugin,
  legendAsTooltipPlugin,
  candlestickPlugin,
  touchZoomPlugin,
  wheelZoomPanPlugin,
} from "@/chart-plugins";

function fmtUSD(val: number, dec: number) {
  return "$" + val.toFixed(dec).replace(/\d(?=(\d{3})+(?:\.|$))/g, "$&,");
}

interface PriceData {
  ticker: string;
  timestamp: Date;
  ts: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

type FormattedPriceData = [
  timestamp: number[],
  open: number[],
  high: number[],
  low: number[],
  close: number[],
  volume: number[]
];

const convertData = (data: PriceData[]): FormattedPriceData =>
  data.reduce(
    (a, e) => {
      a[0].push(new Date(e.ts).getTime() / 1000);
      a[1].push(e.open);
      a[2].push(e.high);
      a[3].push(e.low);
      a[4].push(e.close);
      a[5].push(e.volume);

      return a;
    },
    [[], [], [], [], [], []] as FormattedPriceData
  );

const DATA_INTERVALS = [1, 5, 10, 15, 30, 60, 360, 720, 1080, 1440];

async function loadData(
  u: uPlot,
  min: number,
  max: number,
  resetScales = true
) {
  const numCandles = 250;

  // const min = u.posToVal(u.select.left, "x");
  // const max = u.posToVal(u.select.left + u.select.width, "x");
  // const { min, max } = u.scales.x;
  console.log(min, max);
  const dateRange = {
    min: new Date(min * 1e3),
    max: new Date(max * 1e3),
  };

  const rangeMinutes = (max - min) / 60;
  const validIntervals = DATA_INTERVALS.filter(
    (i) => rangeMinutes <= numCandles * i
  );

  const resolution = validIntervals[0];

  console.log(
    "Fetching data for range...",
    dateRange,
    rangeMinutes,
    resolution
  );
  const pricePromise = api.service("prices").get("BTCUSD", {
    query: {
      timestamp: { $gt: dateRange.min, $lt: dateRange.max },
      resolution,
    },
  });

  useProgress().attach(pricePromise);

  const prices = convertData(await pricePromise);
  console.log("prices", prices);

  // set new data
  u.setData(prices, resetScales);
}

let dailyPrices = shallowRef<FormattedPriceData | null>(null);

let rootWindow = ref([1, 1]);
let wavelength = ref([0, 0]);

onMounted(async () => {
  const pricePromise = api.service("prices").get("BTCUSD", {
    query: {
      timestamp: { $gt: "2020-12-01", $lt: "2021-02-01" },
      resolution: 60 * 24,
    },
  });

  useProgress().attach(pricePromise);

  const prices = convertData(await pricePromise);
  console.log("prices", prices);
  dailyPrices.value = prices;
});

const options: uPlot.Options = {
  title: "Yolo Swaggin's",
  width: document.body.clientWidth,
  height: 600,
  plugins: [
    columnHighlightPlugin(),
    legendAsTooltipPlugin(),
    candlestickPlugin(),
    touchZoomPlugin({ loadData }),
    wheelZoomPanPlugin({ loadData }),
  ],
  cursor: { drag: { x: true, y: true, uni: 20, dist: 10 } },
  scales: {
    x: {
      // distr: 2,
    },
    vol: {
      distr: 0.1,
      range(
        self: uPlot,
        initMin: number,
        initMax: number,
        scaleKey: string
      ): uPlot.Range.MinMax {
        return [initMin, initMax * 5];
      },
    },
  },
  series: [
    {
      label: "Date",
      value: (u: uPlot, ts: number) => new Date(ts * 1e3).toISOString(), //fmtDate(tzDate(ts)),
    },
    {
      label: "Open",
      value: (u: uPlot, v: number) => fmtUSD(v, 2),
    },
    {
      label: "High",
      value: (u: uPlot, v: number) => fmtUSD(v, 2),
    },
    {
      label: "Low",
      value: (u: uPlot, v: number) => fmtUSD(v, 2),
    },
    {
      label: "Close",
      value: (u: uPlot, v: number) => fmtUSD(v, 2),
    },
    {
      label: "Volume",
      scale: "vol",
    },
  ],
  axes: [
    {},
    {
      values: (u, vals) => vals.map((v) => fmtUSD(v, 0)),
    },
    {
      side: 1,
      scale: "vol",
      grid: { show: false },
    },
  ],
  hooks: {
    init: [
      (u: uPlot) => {
        u.over.ondblclick = (e) => {
          if (dailyPrices.value != null) {
            console.log("Fetching data for full range");

            u.setData(dailyPrices.value);
          }
        };
      },
    ],
    setSelect: [
      (u: uPlot) => {
        const min = u.posToVal(u.select.left, "x");
        const max = u.posToVal(u.select.left + u.select.width, "x");
        return loadData(u, min, max);
      },
    ],
  },
};

function onCreate(chart: uPlot) {
  console.log("Created from render fn");
}
function onDelete(chart: uPlot) {
  console.log("Deleted from render fn");
}
</script>

<template>
  <vue3-progress-bar></vue3-progress-bar>

  <Uplot
    v-if="dailyPrices != null"
    key="render-key"
    :data="dailyPrices"
    :options="options"
    @delete="onDelete"
    @create="onCreate"
  />
</template>
