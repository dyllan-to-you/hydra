import uPlot from "uplot";

const debounce = (callback: (...x: any[]) => void, wait: number) => {
  let timeoutId: number = null;
  return (...args: any[]) => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      callback(...args);
    }, wait);
  };
};

export function columnHighlightPlugin({
  className,
  style = { backgroundColor: "rgba(51,204,255,0.3)" },
} = {}) {
  let underEl, overEl, highlightEl, currIdx;

  function init(u) {
    underEl = u.under;
    overEl = u.over;

    highlightEl = document.createElement("div");

    className && highlightEl.classList.add(className);

    uPlot.assign(highlightEl.style, {
      pointerEvents: "none",
      display: "none",
      position: "absolute",
      left: 0,
      top: 0,
      height: "100%",
      ...style,
    });

    underEl.appendChild(highlightEl);

    // show/hide highlight on enter/exit
    overEl.addEventListener("mouseenter", () => {
      highlightEl.style.display = null;
    });
    overEl.addEventListener("mouseleave", () => {
      highlightEl.style.display = "none";
    });
  }

  function update(u) {
    if (currIdx !== u.cursor.idx) {
      currIdx = u.cursor.idx;

      const [iMin, iMax] = u.series[0].idxs;

      const dx = iMax - iMin;
      const width = u.bbox.width / dx / devicePixelRatio;
      const xVal = u.scales.x.distr == 2 ? currIdx : u.data[0][currIdx];
      const left = u.valToPos(xVal, "x") - width / 2;

      highlightEl.style.transform = "translateX(" + Math.round(left) + "px)";
      highlightEl.style.width = Math.round(width) + "px";
    }
  }

  return {
    opts: (u, opts) => {
      uPlot.assign(opts, {
        cursor: {
          x: false,
          y: false,
        },
      });
    },
    hooks: {
      init: init,
      setCursor: update,
    },
  };
}

// converts the legend into a simple tooltip
export function legendAsTooltipPlugin({
  className,
  style = { backgroundColor: "rgba(255, 249, 196, 0.92)", color: "black" },
} = {}) {
  // TODO: Draw better tooltips for candles and predictions
  let legendEl;

  function init(u, opts) {
    legendEl = u.root.querySelector(".u-legend");

    legendEl.classList.remove("u-inline");
    className && legendEl.classList.add(className);

    uPlot.assign(legendEl.style, {
      textAlign: "left",
      pointerEvents: "none",
      display: "none",
      position: "absolute",
      left: 0,
      top: 0,
      zIndex: 100,
      boxShadow: "2px 2px 10px rgba(0,0,0,0.5)",
      ...style,
    });

    // hide series color markers
    const idents = legendEl.querySelectorAll(".u-marker");

    for (let i = 0; i < idents.length; i++) idents[i].style.display = "none";

    const overEl = u.over;
    overEl.style.overflow = "visible";

    // move legend into plot bounds
    overEl.appendChild(legendEl);

    // show/hide tooltip on enter/exit
    overEl.addEventListener("mouseenter", () => {
      legendEl.style.display = null;
    });
    overEl.addEventListener("mouseleave", () => {
      legendEl.style.display = "none";
    });

    // let tooltip exit plot
    //	overEl.style.overflow = "visible";
  }

  function update(u) {
    const { left, top } = u.cursor;
    legendEl.style.transform = "translate(" + left + "px, " + top + "px)";
  }

  return {
    hooks: {
      init: init,
      setCursor: update,
    },
  };
}

// draws candlestick symbols (expects data in OHLC order)
export function candlestickPlugin({
  gap = 2,
  shadowColor = "#000000",
  bearishColor = "#e54245",
  bullishColor = "#4ab650",
  bodyMaxWidth = 20,
  shadowWidth = 2,
  bodyOutline = 1,
} = {}) {
  function drawCandles(u: uPlot) {
    u.ctx.save();

    const offset = (shadowWidth % 2) / 2;

    u.ctx.translate(offset, offset);

    const [iMin, iMax] = u.series[0].idxs;

    const vol0AsY = u.valToPos(0, "vol", true);

    for (let i = iMin; i <= iMax; i++) {
      const xVal = u.scales.x.distr == 2 ? i : u.data[0][i];
      const open = u.data[1][i];
      const high = u.data[2][i];
      const low = u.data[3][i];
      const close = u.data[4][i];
      const vol = u.data[5][i];
      const extrapolated = u.data[6][i];
      const extrapolatedIsup = u.data[7][i];
      // const extrapolatedWavelength = u.data[8][i];
      // const extrapolatedAmplitude = u.data[9][i];
      const predictionMade = u.data[10][i];
      // const projection = u.data[11][i];

      if (xVal != null && open != null) {
        const timeAsX = u.valToPos(xVal, "x", true);
        const lowAsY = u.valToPos(low, "y", true);
        const highAsY = u.valToPos(high, "y", true);
        const openAsY = u.valToPos(open, "y", true);
        const closeAsY = u.valToPos(close, "y", true);
        const volAsY = u.valToPos(vol, "vol", true);
        const extrapolatedAsY = u.valToPos(extrapolated, "y", true);

        // shadow rect
        const shadowHeight =
          Math.max(highAsY, lowAsY) - Math.min(highAsY, lowAsY);
        const shadowX = timeAsX - shadowWidth / 2;
        const shadowY = Math.min(highAsY, lowAsY);

        u.ctx.fillStyle = shadowColor;
        u.ctx.fillRect(
          Math.round(shadowX),
          Math.round(shadowY),
          Math.round(shadowWidth),
          Math.round(shadowHeight)
        );

        // body rect
        const columnWidth = u.bbox.width / (iMax - iMin);
        const bodyWidth = Math.min(bodyMaxWidth, columnWidth - gap);
        const bodyHeight =
          Math.max(closeAsY, openAsY) - Math.min(closeAsY, openAsY);
        const bodyX = timeAsX - bodyWidth / 2;
        const bodyY = Math.min(closeAsY, openAsY);
        const bodyColor = open > close ? bearishColor : bullishColor;

        u.ctx.fillStyle = shadowColor;
        u.ctx.fillRect(
          Math.round(bodyX),
          Math.round(bodyY),
          Math.round(bodyWidth),
          Math.round(bodyHeight)
        );

        u.ctx.fillStyle = bodyColor;
        u.ctx.fillRect(
          Math.round(bodyX + bodyOutline),
          Math.round(bodyY + bodyOutline),
          Math.round(bodyWidth - bodyOutline * 2),
          Math.round(bodyHeight - bodyOutline * 2)
        );

        // volume rect
        u.ctx.fillStyle = bodyColor + "40";
        u.ctx.fillRect(
          Math.round(bodyX),
          Math.round(volAsY),
          Math.round(bodyWidth),
          Math.round(vol0AsY - volAsY)
        );

        if (extrapolated != null) {
          // extrapolated marker
          // the triangle
          const triangleHeight = extrapolatedIsup ? -bodyWidth : bodyWidth;
          u.ctx.beginPath();
          u.ctx.moveTo(timeAsX, extrapolatedAsY);
          u.ctx.lineTo(bodyX, extrapolatedAsY + triangleHeight);
          u.ctx.lineTo(bodyX + bodyWidth, extrapolatedAsY + triangleHeight);
          u.ctx.closePath();

          // the outline
          u.ctx.lineWidth = bodyOutline;
          u.ctx.strokeStyle = shadowColor;
          u.ctx.stroke();

          /**
          take projection and put into buckets. map buckets to color
          when drawing triangles, use bucket color as fill
          */

          // the fill color
          u.ctx.fillStyle = bodyColor;
          u.ctx.fill();

          // weird line thing
          u.ctx.beginPath();
          u.ctx.moveTo(timeAsX, extrapolatedAsY);
          u.ctx.lineTo(u.valToPos(predictionMade, "x", true), extrapolatedAsY);
          u.ctx.closePath();
          // console.log(
          //   "wat",
          //   timeAsX,
          //   extrapolatedAsY,
          //   u.valToPos(predictionMade, "x", true),
          //   extrapolatedAsY
          // );
          u.ctx.lineWidth = 2;
          u.ctx.strokeStyle = shadowColor;
          u.ctx.stroke();
        }
      }
    }

    u.ctx.translate(-offset, -offset);

    u.ctx.restore();
  }

  return {
    opts: (u, opts) => {
      uPlot.assign(opts, {
        cursor: {
          points: {
            show: false,
          },
        },
      });

      opts.series.forEach((series, i) => {
        if (i > 5) return;
        series.paths = () => null;
        series.points = { show: false };
      });
    },
    hooks: {
      draw: drawCandles,
    },
  };
}

interface Position {
  x: number;
  y: number;
  dx: number;
  dy: number;
  d: number;
}

export function touchZoomPlugin(opts: Record<string, any> = {}) {
  function init(u: uPlot) {
    const over = u.over;
    let rect = over.getBoundingClientRect();
    let oxRange, oyRange, xVal, yVal;
    const fr: Position = { x: 0, y: 0, dx: 1, dy: 1, d: 1 };
    const to: Position = { x: 0, y: 0, dx: 1, dy: 1, d: 1 };

    function storePos(t: Position, e: TouchEvent) {
      const ts = e.touches;

      const t0 = ts[0];
      const t0x = t0.clientX - rect.left;
      const t0y = t0.clientY - rect.top;

      // one touch = pan
      if (ts.length == 1) {
        t.x = t0x;
        t.y = t0y;
        t.d = t.dx = t.dy = 1;

        // 2+ touch = zoom
      } else {
        const t1 = e.touches[1];
        const t1x = t1.clientX - rect.left;
        const t1y = t1.clientY - rect.top;

        const xMin = Math.min(t0x, t1x);
        const yMin = Math.min(t0y, t1y);
        const xMax = Math.max(t0x, t1x);
        const yMax = Math.max(t0y, t1y);

        // midpts
        t.y = (yMin + yMax) / 2;
        t.x = (xMin + xMax) / 2;

        t.dx = xMax - xMin;
        t.dy = yMax - yMin;

        // dist
        t.d = Math.sqrt(t.dx * t.dx + t.dy * t.dy);
      }
    }

    let rafPending = false;

    function zoom() {
      rafPending = false;

      const left = to.x;
      const top = to.y;

      // non-uniform scaling
      //	let xFactor = fr.dx / to.dx;
      //	let yFactor = fr.dy / to.dy;

      // uniform x/y scaling
      const xFactor = fr.d / to.d;
      const yFactor = fr.d / to.d;

      const leftPct = left / rect.width;
      const btmPct = 1 - top / rect.height;

      const nxRange = oxRange * xFactor;
      const nxMin = xVal - leftPct * nxRange;
      const nxMax = nxMin + nxRange;

      const nyRange = oyRange * yFactor;
      const nyMin = yVal - btmPct * nyRange;
      const nyMax = nyMin + nyRange;

      u.batch(() => {
        u.setScale("x", {
          min: nxMin,
          max: nxMax,
        });

        u.setScale("y", {
          min: nyMin,
          max: nyMax,
        });
      });
    }

    function touchmove(e: TouchEvent) {
      storePos(to, e);

      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(zoom);
      }
    }

    over.addEventListener("touchstart", function (e) {
      rect = over.getBoundingClientRect();

      storePos(fr, e);

      oxRange = u.scales.x.max - u.scales.x.min;
      oyRange = u.scales.y.max - u.scales.y.min;

      const left = fr.x;
      const top = fr.y;

      xVal = u.posToVal(left, "x");
      yVal = u.posToVal(top, "y");

      document.addEventListener("touchmove", touchmove, { passive: true });
    });

    over.addEventListener("touchend", function (e: TouchEvent) {
      document.removeEventListener("touchmove", touchmove);
      if (opts.loadData) {
        opts.loadData({ start: u.scales.x.min, end: u.scales.x.max }, u);
      }
    });
  }

  return {
    hooks: {
      init,
    },
  };
}

enum MouseButton { // https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button
  Left = 0,
  Primary = 0,
  Middle = 1,
  Auxiliary = 1,
  Right = 2,
  Secondary = 2,
  Fourth = 3,
  BrowserBack = 3,
  Fifth = 4,
  BrowserForward = 4,
}

export function wheelZoomPanPlugin(opts: Record<string, any> = {}) {
  opts = { factor: 0.75, mouseButton: MouseButton.Middle, ...opts };
  const factor = opts.factor || 0.75;

  let xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    xRange: number,
    yRange: number;

  function clamp(
    nRange: number,
    nMin: number,
    nMax: number,
    fRange: number,
    fMin: number,
    fMax: number
  ) {
    if (nRange > fRange) {
      nMin = fMin;
      nMax = fMax;
    } else if (nMin < fMin) {
      nMin = fMin;
      nMax = fMin + nRange;
    } else if (nMax > fMax) {
      nMax = fMax;
      nMin = fMax - nRange;
    }

    return [nMin, nMax];
  }

  return {
    hooks: {
      ready: (u: uPlot) => {
        xMin = u.scales.x.min;
        xMax = u.scales.x.max;
        yMin = u.scales.y.min;
        yMax = u.scales.y.max;

        xRange = xMax - xMin;
        yRange = yMax - yMin;

        const over = u.over;
        const rect = over.getBoundingClientRect();

        const wheelDebouncer = debounce(() => {
          console.log("debouncey");
          opts.loadData({ start: u.scales.x.min, end: u.scales.x.max }, u);
        }, 50);

        // wheel drag pan
        over.addEventListener("mousedown", (e) => {
          if (e.button == opts.mouseButton) {
            //	plot.style.cursor = "move";
            e.preventDefault();

            const left0 = e.clientX;
            //	let top0 = e.clientY;

            const scXMin0 = u.scales.x.min;
            const scXMax0 = u.scales.x.max;

            const xUnitsPerPx = u.posToVal(1, "x") - u.posToVal(0, "x");

            const onmove = (e) => {
              e.preventDefault();

              const left1 = e.clientX;
              //	let top1 = e.clientY;

              const dx = xUnitsPerPx * (left1 - left0);

              u.setScale("x", {
                min: scXMin0 - dx,
                max: scXMax0 - dx,
              });
            };

            const onup = (e) => {
              document.removeEventListener("mousemove", onmove);
              document.removeEventListener("mouseup", onup);
              if (opts.loadData) {
                opts.loadData(
                  { start: u.scales.x.min, end: u.scales.x.max },
                  u
                );
              }
            };

            document.addEventListener("mousemove", onmove);
            document.addEventListener("mouseup", onup);
          }
        });

        // wheel scroll zoom
        over.addEventListener("wheel", (e) => {
          e.preventDefault();

          const { left, top } = u.cursor;

          const leftPct = left / rect.width;
          const btmPct = 1 - top / rect.height;
          const xVal = u.posToVal(left, "x");
          const yVal = u.posToVal(top, "y");
          const oxRange = u.scales.x.max - u.scales.x.min;
          const oyRange = u.scales.y.max - u.scales.y.min;

          const nxRange = e.deltaY < 0 ? oxRange * factor : oxRange / factor;
          let nxMin = xVal - leftPct * nxRange;
          let nxMax = nxMin + nxRange;
          console.log(
            "wheelin",
            nxRange,
            [nxMin, nxMax],
            clamp(nxRange, nxMin, nxMax, xRange, xMin, xMax),
            xRange,
            xMin,
            xMax
          );
          // [nxMin, nxMax] = clamp(nxRange, nxMin, nxMax, xRange, xMin, xMax);

          const nyRange = e.deltaY < 0 ? oyRange * factor : oyRange / factor;
          let nyMin = yVal - btmPct * nyRange;
          let nyMax = nyMin + nyRange;
          [nyMin, nyMax] = clamp(nyRange, nyMin, nyMax, yRange, yMin, yMax);

          u.batch(() => {
            u.setScale("x", {
              min: nxMin,
              max: nxMax,
            });

            u.setScale("y", {
              min: nyMin,
              max: nyMax,
            });
            if (opts.loadData) {
              wheelDebouncer();
            }
          });
        });
      },
    },
  };
}
