// Envoltura de ECharts: toma los colores del tema (variables CSS) y se redibuja al cambiar de tamaño o de tema.
import { BarChart, BoxplotChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent,
  VisualMapComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef, useState } from "react";

echarts.use([BarChart, BoxplotChart, HeatmapChart, LineChart, ScatterChart, GridComponent, LegendComponent, MarkAreaComponent,
  MarkLineComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export interface Theme {
  surface: string;
  ink: string;
  ink2: string;
  ink3: string;
  rule: string;
  series: [string, string, string];
  up: string;
  down: string;
  point: string;
  mutedBar: string;
  mark: string;
  action: string;
  font: string;
}

function readTheme(): Theme {
  const style = getComputedStyle(document.documentElement);
  const value = (name: string) => style.getPropertyValue(name).trim();
  return {
    surface: value("--surface"), ink: value("--ink"), ink2: value("--ink-2"), ink3: value("--ink-3"),
    rule: value("--rule"), series: [value("--series-1"), value("--series-2"), value("--series-3")],
    up: value("--up"), down: value("--down"), point: value("--point"), mutedBar: value("--muted-bar"),
    mark: value("--mark"), action: value("--action"), font: value("--font-ui"),
  };
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(readTheme());
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return theme;
}

// Ejes, cuadrícula y tooltip comunes, con tinta recesiva para que resalten los datos
export function axis(theme: Theme, extra: Record<string, unknown> = {}) {
  return {
    axisLine: { lineStyle: { color: theme.rule } },
    axisTick: { show: false },
    axisLabel: { color: theme.ink3, fontFamily: theme.font, fontSize: 12 },
    splitLine: { lineStyle: { color: theme.rule, opacity: 0.7 } },
    nameTextStyle: { color: theme.ink2, fontFamily: theme.font, fontSize: 13 },
    nameGap: 30,
    ...extra,
  };
}

export function tooltip(theme: Theme, extra: Record<string, unknown> = {}) {
  return {
    backgroundColor: theme.surface,
    borderColor: theme.rule,
    textStyle: { color: theme.ink, fontFamily: theme.font, fontSize: 13 },
    extraCssText: "border-radius:10px;box-shadow:0 6px 24px rgb(0 0 0 / 0.12);",
    ...extra,
  };
}

export function legend(theme: Theme, extra: Record<string, unknown> = {}) {
  return {
    top: 0,
    left: 0,
    icon: "roundRect",
    itemWidth: 12,
    itemHeight: 12,
    textStyle: { color: theme.ink2, fontFamily: theme.font, fontSize: 13 },
    ...extra,
  };
}

interface ChartProps {
  option: (theme: Theme) => echarts.EChartsCoreOption;
  height?: number;
  label: string;
  onClick?: (params: { name?: string; dataIndex?: number; seriesName?: string }) => void;
}

export function Chart({ option, height = 360, label, onClick }: ChartProps) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!element.current) return;
    const instance = echarts.init(element.current, null, { renderer: "canvas" });
    chart.current = instance;
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(
      { animationDuration: 400, textStyle: { fontFamily: theme.font }, ...option(theme) }, { notMerge: true },
    );
  }, [option, theme]);

  useEffect(() => {
    const instance = chart.current;
    if (!instance || !onClick) return;
    instance.on("click", onClick);
    return () => {
      instance.off("click", onClick);
    };
  }, [onClick]);

  return <div ref={element} role="img" aria-label={label} style={{ width: "100%", height }} />;
}
