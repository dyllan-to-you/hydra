<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from "vue";
import { useProgress } from "@marcoschulte/vue3-progress";
import * as dfd from "danfojs";

import type uPlot from "uplot";
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

import type { Data as FFTIndicator } from "@/../../server/services/fft-indicator/fft-indicator.class";

function fmtUSD(val: number, dec: number) {
  return val != null
    ? "$" + val.toFixed(dec).replace(/\d(?=(\d{3})+(?:\.|$))/g, "$&,")
    : "";
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

// Starting or full timespan
let originalTimespan = {
  start: new Date("2017-12-01"),
  end: new Date("2018-02-01"),
};
let currentTimespan = ref({ ...originalTimespan });

let theChart: uPlot | null = null;
let df: dfd.DataFrame | null = null;

let dataset = shallowRef<number[][] | null>(null);
const loadedData = ref(false);

// let availableParameters = computed(() => {
//   currentTimespan
// })

let rootWindow = ref([1, 1]);
let rootWindowRange = computed(() => {
  if (rootWindow.value[1] <= rootWindow.value[0]) {
    return rootWindow.value[0];
  } else {
    return { $gte: rootWindow.value[0], $lte: rootWindow.value[1] };
  }
});

let wavelength = ref([0, 0]);
let wavelengthRange = computed(() => {
  if (wavelength.value[1] <= wavelength.value[0]) {
    return wavelength.value[0];
  } else {
    return { $gte: wavelength.value[0], $lte: wavelength.value[1] };
  }
});

const DATA_INTERVALS = [1, 5, 10, 15, 30, 60, 360, 720, 1440, 4320, 10080];

async function loadData(
  timespan: {
    start: number | string | Date;
    end: number | string | Date;
  } = currentTimespan.value,
  u: uPlot | null = null,
  resetScales = true
) {
  const numCandles = 250;
  const { start, end } = timespan;

  const dateRange = {
    start: new Date(typeof start === "number" ? start * 1e3 : start),
    end: new Date(typeof end === "number" ? end * 1e3 : end),
  };
  currentTimespan.value = dateRange;

  const rangeMinutes =
    (dateRange.end.getTime() - dateRange.start.getTime()) / 1000 / 60;
  const validIntervals = DATA_INTERVALS.filter(
    (i) => rangeMinutes <= numCandles * i
  );

  const resolution = validIntervals[0];

  console.log(
    "Fetching data for range...",
    dateRange,
    rangeMinutes,
    resolution,
    rootWindowRange.value,
    wavelengthRange.value
  );
  const pricePromise = api.service("prices").get("BTCUSD", {
    query: {
      timestamp: { $gt: dateRange.start, $lt: dateRange.end },
      resolution,
    },
  });

  const fftPromise: Promise<FFTIndicator[]> = api
    .service("fft-indicator")
    .find({
      query: {
        first_extrapolated_date: { $gt: dateRange.start, $lt: dateRange.end },
        rootNumber: rootWindowRange.value,
        ifft_extrapolated_wavelength: wavelengthRange.value,
      },
    });

  useProgress().attach(pricePromise);
  useProgress().attach(fftPromise);

  const priceDf = new dfd.DataFrame(
    (await pricePromise).map((e) => {
      return {
        ...e,
        ts: new Date(e.ts).getTime() / 1000,
      };
    })
  );
  console.log("prices", priceDf);

  const fftDf = new dfd.DataFrame(
    (await fftPromise).map((e) => {
      const ts = new Date(e.first_extrapolated_date).getTime() / 1000;
      const predictionMade = new Date(e.endDate).getTime() / 1000;
      return {
        ...e,
        ts,
        predictionMade,
        projection: ts - predictionMade,
      };
    }),
    {
      columns: [
        "minPerCycle",
        "deviance",
        "ifft_extrapolated_wavelength",
        "ifft_extrapolated_amplitude",
        "ifft_extrapolated_deviance",
        "first_extrapolated",
        "first_extrapolated_date",
        "first_extrapolated_isup",
        "startDate",
        "endDate",
        "window",
        "window_original",
        "trend_deviance",
        "trend_slope",
        "trend_intercept",
        "rootNumber",
        "ts", // `first_extrapolated_date` as unix timestamp
        "predictionMade", // `endDate` as unix timestamp
        "projection", // How far into the future? (first_extrapolated_date - endDate)
      ],
    }
  );
  console.log("fft", fftDf);

  df = dfd.merge({
    left: priceDf,
    right: fftDf,
    on: ["ts"],
    how: "outer",
  });
  console.log("merged", df);

  /*
  outputs["text"] = (
      outputs["endDate"].astype(str)
      + "("
      + (outputs["first_extrapolated_date"] - outputs["endDate"]).astype(str)
      + ")<br>isup = "
      + outputs["first_extrapolated_isup"].astype(str)
      + "<br>🌊"
      + outputs["ifft_extrapolated_wavelength"].astype(str)
      + "<br>🔊"
      + outputs["ifft_extrapolated_amplitude"].astype(str)
  )
  */

  // set new data
  dataset.value = df
    .loc({
      columns: [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "first_extrapolated",
        "first_extrapolated_isup",
        "ifft_extrapolated_wavelength",
        "ifft_extrapolated_amplitude",
        "predictionMade",
        "projection",
      ],
    })
    .getColumnData.map((series) =>
      series.map((val) => (isNaN(val) ? null : val))
    );

  if (u != null && dataset.value) {
    console.log("uPlot setdata");
    u.setData(dataset.value, resetScales);
  } else if (theChart != null) {
    console.log("Reference setdata");
    theChart.setData(dataset.value);
  }

  loadedData.value = true;
  return dataset.value;
}

// let initialLoad = ref(false);
onMounted(async () => {
  await loadData(originalTimespan);
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
    {
      label: "Extrapolated",
      value: (u: uPlot, v: number) => fmtUSD(v, 2),
      spanGaps: false,
      // points: {
      //   show: true,
      //   stroke: "blue",
      // },
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
          loadData(originalTimespan, u);
          // if (dailyPrices.value != null) {
          //   console.log("Fetching data for full range");

          //   u.setData(dailyPrices.value);
          // }
        };
      },
    ],
    setSelect: [
      (u: uPlot) => {
        const start = u.posToVal(u.select.left, "x");
        const end = u.posToVal(u.select.left + u.select.width, "x");
        return loadData({ start, end }, u);
      },
    ],
  },
};

function onCreate(chart: uPlot) {
  console.log("Created from render fn");
  theChart = chart;
}
function onDelete(chart: uPlot) {
  console.log("Deleted from render fn");
}
</script>

<template>
  <vue3-progress-bar></vue3-progress-bar>

  <u-plot
    v-if="loadedData"
    key="render-key"
    :data="dataset"
    :options="options"
    @delete="onDelete"
    @create="onCreate"
  />
  <h3>Root Window: {{ rootWindow }}</h3>
  <vueform-slider v-model="rootWindow"></vueform-slider>
  <h3>Wavelength: {{ wavelength }}</h3>
  <vueform-slider v-model="wavelength"></vueform-slider>
</template>
