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

let rootWindowStart = ref(0);
let rootWindowEnd = ref(1);
let rootWindow = computed({
  set: ([start, end]) => {
    rootWindowStart.value = start;
    rootWindowEnd.value = end;
  },
  get: () => [rootWindowStart.value, rootWindowEnd.value],
});
let rootWindowRange = computed(() => {
  if (rootWindow.value[1] <= rootWindow.value[0]) {
    return rootWindow.value[0];
  } else {
    return { $gte: rootWindow.value[0], $lte: rootWindow.value[1] };
  }
});

let wavelengthStart = ref(0);
let wavelengthEnd = ref(1);
let wavelengthLog10 = computed({
  set: ([start, end]) => {
    wavelengthStart.value = 10 ** start;
    wavelengthEnd.value = 10 ** end;
  },
  get: () => [
    Math.log10(wavelengthStart.value),
    Math.log10(wavelengthEnd.value),
  ],
});
let wavelengthRange = computed(() => {
  if (wavelengthEnd.value <= wavelengthStart.value) {
    return wavelengthStart.value;
  } else {
    return { $gte: wavelengthStart.value, $lte: wavelengthEnd.value };
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
    "rootWindowRange",
    rootWindowRange.value,
    "wavelengthRange",
    wavelengthRange.value
  );
  const pricePromise = api.service("prices").get("BTCUSD", {
    query: {
      timestamp: { $gt: dateRange.start, $lt: dateRange.end },
      resolution,
    },
  });

  const fftInfoPromise: Promise<FFTIndicator[]> = api
    .service("fft-indicator")
    .get("info", {
      query: {
        first_extrapolated_date: { $gt: dateRange.start, $lt: dateRange.end },
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
  useProgress().attach(fftInfoPromise);
  useProgress().attach(fftPromise);

  fftInfo.value = await fftInfoPromise;
  console.log("INFO", fftInfo.value);

  const priceDf = new dfd.DataFrame(
    (await pricePromise).map((e) => {
      return {
        ...e,
        ts: new Date(e.ts).getTime() / 1000,
      };
    })
  );
  console.log("prices", priceDf);
  console.log(
    "fftPromise",
    "full",
    await api.service("fft-indicator").find({
      query: {
        rootNumber: rootWindowRange.value,
        ifft_extrapolated_wavelength: wavelengthRange.value,
      },
    }),
    "partial",
    await fftPromise
  );
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

const fftInfo = ref({ rootNumbers: [], wavelengths: [] });
const fftMetaInfo = ref({
  rootNumbers: { min: 0, max: 0 },
  wavelengths: { min: 0, max: 0 },
});

// let initialLoad = ref(false);
onMounted(async () => {
  await loadData(originalTimespan);
  const results = await api.service("fft-indicator").get("info", {
    query: {},
  });
  console.log("Initializing metainfo", results);
  fftMetaInfo.value.rootNumbers = {
    min: Math.min(...results.rootNumbers),
    max: Math.max(...results.rootNumbers),
  };
  fftMetaInfo.value.wavelengths = {
    min: Math.min(...results.wavelengths),
    max: Math.max(...results.wavelengths),
  };
  console.log("Initialized metainfo", fftMetaInfo.value);
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

function test(x) {
  console.log("test", x);
  return x;
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
  <v-card>
    <v-card-title
      >Root Window: {{ rootWindow[0] }} - {{ rootWindow[1] }}</v-card-title
    >
    <v-card-text>
      <v-row>
        <v-range-slider
          v-model="rootWindow"
          :min="fftMetaInfo.rootNumbers?.min"
          :max="fftMetaInfo.rootNumbers?.max"
          step="1"
          :ticks="fftInfo.rootNumbers"
          show-ticks="always"
          tick-size="5"
          track-size="12"
          strict
        >
          <template v-slot:prepend>
            <v-text-field
              v-model="rootWindowStart"
              hide-details
              single-line
              type="number"
              variant="outlined"
              density="compact"
              :error="!fftInfo.rootNumbers.includes(rootWindowStart)"
            ></v-text-field>
          </template>
          <template v-slot:append>
            <v-text-field
              v-model="rootWindowEnd"
              hide-details
              single-line
              type="number"
              variant="outlined"
              density="compact"
              :error="!fftInfo.rootNumbers.includes(rootWindow[1])"
            ></v-text-field>
          </template>
        </v-range-slider>
      </v-row>
    </v-card-text>
  </v-card>
  <v-card>
    <v-card-title>
      Wavelength:
      {{ wavelengthStart }} -
      {{ wavelengthEnd }}
    </v-card-title>
    <v-card-text>
      <v-row>
        <v-range-slider
          v-model="wavelengthLog10"
          :min="Math.log10(fftMetaInfo.wavelengths?.min || 10)"
          :max="Math.log10(fftMetaInfo.wavelengths?.max)"
          :ticks="fftInfo.wavelengths.map((e) => Math.log10(e || 10))"
          show-ticks="always"
          tick-size="5"
          track-size="12"
          strict
        >
          <template v-slot:prepend>
            <v-text-field
              v-model.number="wavelengthStart"
              :error="!fftInfo.wavelengths.includes(wavelengthStart)"
              hide-details
              single-line
              type="number"
              variant="outlined"
              density="compact"
            ></v-text-field>
          </template>
          <template v-slot:append>
            <v-text-field
              v-model.number="wavelengthEnd"
              :error="!fftInfo.wavelengths.includes(wavelengthEnd)"
              hide-details
              single-line
              type="number"
              variant="outlined"
              density="compact"
            ></v-text-field>
          </template>
        </v-range-slider>
      </v-row>
    </v-card-text>
  </v-card>
</template>
