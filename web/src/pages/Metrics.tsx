// Métricas: rendimiento de cada algoritmo en los datos de prueba (validación dejando un prompt fuera),
// con gráficas interactivas que el usuario puede mostrar u ocultar.
import { useCallback, useMemo, useState } from "react";

import { type Modelo, type ModeloPorPrompt, type PruebaOptuna, type Resumenes, type Target, TARGETS, useApi } from "../api";
import { axis, Chart, legend, type Theme, tooltip } from "../components/Chart";
import { WarnIcon } from "../components/Dropzone";
import { boxStats, fixed, mean, rmse } from "../stats";

// Rampa secuencial viridis: va de claro a oscuro sin cambiar de orden para ningún tipo de daltonismo
const HEAT_RAMP = ["#FDE725", "#7AD151", "#22A884", "#2A788E", "#414487", "#440154"];

// Colores fijos por papel, no por familia: con 7 familias ninguna paleta se distingue bien con daltonismo
function roleColor(theme: Theme, model: Modelo) {
  if (model.es_modelo_aplicacion) return theme.series[0];
  if (model.es_mejor_modelo) return theme.series[1];
  return theme.series[2];
}

interface ModelosResponse {
  modelos: Modelo[];
  por_prompt: ModeloPorPrompt[];
}

const short = (name: string, size = 34) => (name.length > size ? `${name.slice(0, size - 1)}…` : name);

export function Metrics() {
  const { data, error } = useApi<ModelosResponse>("/api/modelos");
  const [showCharts, setShowCharts] = useState(true);
  const [showTable, setShowTable] = useState(true);
  const [family, setFamily] = useState("Todos");
  const [open, setOpen] = useState<Set<string>>(new Set());

  const models = useMemo(() => [...(data?.modelos ?? [])].sort((a, b) => a.mcrmse - b.mcrmse), [data]);
  const families = ["Todos", ...new Set(models.map((model) => model.familia))];
  const visible = models.filter((model) => family === "Todos" || model.familia === family);
  const app = models.find((model) => model.es_modelo_aplicacion);
  const best = models.find((model) => model.es_mejor_modelo);
  const dummy = models.find((model) => model.modelo.startsWith("Dummy"));

  // Al abrir la app, el modelo en uso aparece desplegado
  const isOpen = (name: string) => open.has(name) || (open.size === 0 && name === app?.modelo && !open.has("__none"));
  const toggle = (name: string) => setOpen((current) => {
    const next = new Set([...current].filter((value) => value !== "__none"));
    if (current.size === 0 && app) next.add(app.modelo);
    if (isOpen(name)) next.delete(name); else next.add(name);
    if (next.size === 0) next.add("__none");
    return next;
  });

  const openFromChart = useCallback((params: { name?: string }) => {
    const model = models.find((m) => short(m.modelo) === params.name || m.modelo === params.name);
    if (!model) return;
    setOpen((current) => new Set([...current, model.modelo].filter((value) => value !== "__none")));
    requestAnimationFrame(() => document.getElementById(`modelo-${models.indexOf(model)}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  }, [models]);

  return (
    <>
      <header className="page-head rise">
        <h1 className="page-title">Métricas del modelo</h1>
        <p>Rendimiento de cada algoritmo que implementamos, medido con
          validación cruzada por prompt: cada modelo se evalúa sobre un texto que no vio al entrenar. En RMSE y MCRMSE,
          menos es mejor; en R², más.</p>
      </header>

      {error && <p className="alert" role="alert"><WarnIcon />No se pudieron cargar las métricas: {error}</p>}

      <div className="kpis">
        <div className="kpi dark"><span className="kicker">Modelo de la app · MCRMSE</span>
          <span className="value">{app ? fixed(app.mcrmse) : "…"}</span><span className="sub">{app?.modelo}</span></div>
        <div className="kpi"><span className="kicker">Mejor modelo · MCRMSE</span>
          <span className="value">{best ? fixed(best.mcrmse) : "…"}</span><span className="sub">{best?.modelo}</span></div>
        <div className="kpi"><span className="kicker">Línea base · MCRMSE</span>
          <span className="value">{dummy ? fixed(dummy.mcrmse) : "…"}</span><span className="sub">Dummy: predice siempre la media</span></div>
      </div>

      <div className="toolbar" role="toolbar" aria-label="Opciones de visualización">
        <button type="button" className="toggle" aria-pressed={showCharts} onClick={() => setShowCharts(!showCharts)}>
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 16V9M9 16V4M14 16v-5" /></svg>
          {showCharts ? "Ocultar gráficas" : "Mostrar gráficas"}
        </button>
        <button type="button" className="toggle" aria-pressed={showTable} onClick={() => setShowTable(!showTable)}>
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="3" y="4" width="14" height="12" rx="2" /><path d="M3 8h14M8 8v8" /></svg>
          {showTable ? "Ocultar tabla" : "Mostrar tabla"}
        </button>
        <span className="sep" aria-hidden="true" />
        <span className="label">Familia</span>
        <div className="pill-group">
          {families.map((name) => (
            <button key={name} type="button" className="pill" aria-pressed={family === name} onClick={() => setFamily(name)}>{name}</button>
          ))}
        </div>
        <div className="toolbar-end">
          <button type="button" className="link-btn" onClick={() => setOpen(new Set([...open, ...visible.map((m) => m.modelo)].filter((v) => v !== "__none")))}>Expandir todos</button>
          <button type="button" className="link-btn" onClick={() => setOpen(new Set(["__none"]))}>Contraer todos</button>
        </div>
      </div>

      {showCharts && data && (
        <section aria-labelledby="ov-h" className="metrics-section loose rise">
          <h2 id="ov-h">Comparación general</h2>
          <div className="chart-grid">
            <figure className="chart-card">
              <figcaption>MCRMSE por modelo<span>clic en una barra para ver su detalle</span></figcaption>
              <McrmseChart models={visible} onClick={openFromChart} />
            </figure>
            <figure className="chart-card">
              <figcaption>RMSE content vs wording<span>un punto por modelo · más cerca del origen, mejor</span></figcaption>
              <ScatterModels models={visible} onClick={openFromChart} />
            </figure>
            <figure className="chart-card">
              <figcaption>MCRMSE por prompt<span>cada prompt se evaluó sin haberlo visto</span></figcaption>
              <PromptHeatmap models={visible} rows={data.por_prompt} />
            </figure>
            <figure className="chart-card">
              <figcaption>Búsqueda de Optuna<span>notebook 11 · búsqueda final con los 4 prompts</span></figcaption>
              <OptunaChart />
            </figure>
          </div>
        </section>
      )}

      {showTable && data && (
        <section aria-labelledby="tb-h" className="metrics-section rise">
          <h2 id="tb-h">Tabla de métricas</h2>
          <div className="table-box">
            <table className="table wide">
              <thead>
                <tr><th scope="col">Modelo</th><th scope="col">Familia</th><th scope="col" className="num">RMSE content</th>
                  <th scope="col" className="num">R² content</th><th scope="col" className="num">RMSE wording</th>
                  <th scope="col" className="num">R² wording</th><th scope="col" className="num">MCRMSE</th></tr>
              </thead>
              <tbody>
                {visible.map((model) => (
                  <tr key={model.modelo} className={model.es_modelo_aplicacion ? "highlight" : ""}>
                    <td>{model.modelo}</td><td className="muted">{model.familia}</td>
                    <td className="num">{fixed(model.rmse_content)}</td><td className="num">{fixed(model.r2_content)}</td>
                    <td className="num">{fixed(model.rmse_wording)}</td><td className="num">{fixed(model.r2_wording)}</td>
                    <td className="num">{fixed(model.mcrmse)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section aria-labelledby="md-h" className="metrics-section">
        <div className="row-between">
          <h2 id="md-h">Detalle por algoritmo</h2>
          <span className="muted small">{visible.length} de {models.length} modelos</span>
        </div>
        {!data && !error && [0, 1, 2, 3].map((i) => <div key={i} className="skeleton row" aria-hidden="true" />)}
        {!data && !error && <p className="sr-only" role="status">Cargando modelos…</p>}
        {visible.map((model) => {
          const expanded = isOpen(model.modelo);
          const index = models.indexOf(model);
          return (
            <article key={model.modelo} id={`modelo-${index}`} className={`model-card${model.es_modelo_aplicacion ? " app" : ""}`}>
              <button type="button" className="model-head" aria-expanded={expanded} aria-controls={`modelo-body-${index}`}
                onClick={() => toggle(model.modelo)}>
                <svg className="chev" width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 4l5 5-5 5" /></svg>
                <span className="model-title">
                  <strong>{model.modelo}</strong>
                  <span>{model.familia} · notebook {model.notebook} · {model.etapa}</span>
                </span>
                {model.es_modelo_aplicacion && <span className="badge dark">En la app</span>}
                {model.es_mejor_modelo && <span className="badge clay">Mejor MCRMSE</span>}
                <span className="model-score">MCRMSE <strong>{fixed(model.mcrmse)}</strong></span>
              </button>
              {expanded && (
                <div id={`modelo-body-${index}`} className="model-body rise">
                  <div className="mini-metrics">
                    <div><small>RMSE content</small><b>{fixed(model.rmse_content)}</b></div>
                    <div><small>R² content</small><b>{fixed(model.r2_content)}</b></div>
                    <div><small>RMSE wording</small><b>{fixed(model.rmse_wording)}</b></div>
                    <div><small>R² wording</small><b>{fixed(model.r2_wording)}</b></div>
                    <div><small>Desv. entre prompts</small><b>{fixed(model.desviacion_entre_prompts)}</b></div>
                  </div>
                  {showCharts && <ModelCharts model={model} byPrompt={data?.por_prompt ?? []} />}
                </div>
              )}
            </article>
          );
        })}
      </section>
    </>
  );
}

function McrmseChart({ models, onClick }: { models: Modelo[]; onClick: (params: { name?: string }) => void }) {
  const option = useCallback((theme: Theme) => {
    const rows = [...models].reverse();
    return {
      grid: { left: 8, right: 48, top: 8, bottom: 24, containLabel: true },
      tooltip: tooltip(theme, { trigger: "axis", axisPointer: { type: "shadow" },
        formatter: (items: { dataIndex: number }[]) => {
          const model = rows[items[0].dataIndex];
          return `<b>${model.modelo}</b><br/>MCRMSE ${fixed(model.mcrmse)}<br/>content ${fixed(model.rmse_content)} · wording ${fixed(model.rmse_wording)}`;
        } }),
      xAxis: { type: "value", min: 0, ...axis(theme) },
      yAxis: { type: "category", data: rows.map((m) => short(m.modelo)), ...axis(theme, { axisLabel: { color: theme.ink2, fontSize: 11, fontFamily: theme.font } }) },
      series: [{
        type: "bar", barWidth: "62%", cursor: "pointer",
        data: rows.map((m) => ({ value: m.mcrmse, itemStyle: { color: roleColor(theme, m), borderRadius: [0, 4, 4, 0] } })),
        label: { show: true, position: "right", formatter: (p: { value: number }) => p.value.toFixed(3), color: theme.ink2, fontFamily: "IBM Plex Mono", fontSize: 11 },
        emphasis: { itemStyle: { borderColor: theme.ink, borderWidth: 1.5 } },
      }],
    };
  }, [models]);
  return (
    <>
      <RoleLegend />
      <Chart option={option} height={Math.max(260, models.length * 26)} label="Barras con el MCRMSE de cada modelo" onClick={onClick} />
    </>
  );
}

function ScatterModels({ models, onClick }: { models: Modelo[]; onClick: (params: { name?: string }) => void }) {
  const option = useCallback((theme: Theme) => {
    const groups = [
      { name: "Modelo de la app", symbol: "circle", size: 18, rows: models.filter((m) => m.es_modelo_aplicacion) },
      { name: "Mejor MCRMSE", symbol: "diamond", size: 20, rows: models.filter((m) => m.es_mejor_modelo && !m.es_modelo_aplicacion) },
      { name: "Otros modelos", symbol: "circle", size: 10, rows: models.filter((m) => !m.es_modelo_aplicacion && !m.es_mejor_modelo) },
    ].filter((group) => group.rows.length);
    return {
      grid: { left: 8, right: 16, top: 40, bottom: 36, containLabel: true },
      legend: legend(theme),
      tooltip: tooltip(theme, { trigger: "item", formatter: (p: { data: { name: string; familia: string; value: number[] } }) =>
        `<b>${p.data.name}</b><br/>${p.data.familia}<br/>RMSE content ${fixed(p.data.value[0])}<br/>RMSE wording ${fixed(p.data.value[1])}` }),
      xAxis: { type: "value", name: "RMSE content", nameLocation: "middle", scale: true, ...axis(theme) },
      yAxis: { type: "value", name: "RMSE wording", nameLocation: "middle", scale: true, ...axis(theme, { nameGap: 40 }) },
      series: groups.map((group, i) => ({
        name: group.name, type: "scatter", cursor: "pointer", symbol: group.symbol, symbolSize: group.size, z: 3 - i,
        itemStyle: { color: roleColor(theme, group.rows[0]), borderColor: theme.surface, borderWidth: 1.5 },
        // Los dos destacados caen juntos abajo a la izquierda: uno se etiqueta arriba y el otro a la derecha
        label: { show: group.name !== "Otros modelos", position: group.name === "Modelo de la app" ? "top" : "right", formatter: group.name, color: theme.ink, fontSize: 12,
          fontWeight: 600, fontFamily: theme.font },
        data: group.rows.map((m) => ({ name: m.modelo, familia: m.familia, value: [m.rmse_content, m.rmse_wording] })),
      })),
    };
  }, [models]);
  return <Chart option={option} height={360} label="Dispersión del RMSE de content contra el de wording por modelo" onClick={onClick} />;
}

function RoleLegend() {
  const items: [string, string][] = [["app", "Modelo de la app"], ["best", "Mejor MCRMSE"], ["other", "Otros modelos"]];
  return (
    <div className="role-legend">
      {items.map(([role, label]) => (
        <span key={label}>
          <span aria-hidden="true" className={`swatch ${role}`} />{label}
        </span>
      ))}
    </div>
  );
}

function PromptHeatmap({ models, rows }: { models: Modelo[]; rows: ModeloPorPrompt[] }) {
  const option = useCallback((theme: Theme) => {
    const names = models.map((m) => m.modelo).filter((name) => rows.some((row) => row.modelo === name)).reverse();
    const prompts = [...new Map(rows.map((row) => [row.prompt_id, row.prompt_title])).entries()];
    const data = rows.filter((row) => names.includes(row.modelo))
      .map((row) => [prompts.findIndex(([id]) => id === row.prompt_id), names.indexOf(row.modelo), Number(row.mcrmse.toFixed(3))]);
    const values = data.map((d) => d[2]);
    return {
      grid: { left: 8, right: 16, top: 8, bottom: 70, containLabel: true },
      tooltip: tooltip(theme, { formatter: (p: { data: number[] }) =>
        `<b>${names[p.data[1]]}</b><br/>${prompts[p.data[0]][1]}: MCRMSE ${p.data[2].toFixed(3)}` }),
      // interval 0: los 4 prompts siempre con nombre; si no caben en una línea, se parten en dos
      xAxis: { type: "category", data: prompts.map(([, title]) => title), ...axis(theme, { splitArea: { show: false },
        axisLabel: { color: theme.ink2, fontSize: 11, fontFamily: theme.font, interval: 0, width: 96, overflow: "break" } }) },
      yAxis: { type: "category", data: names.map((name) => short(name, 28)), ...axis(theme, { axisLabel: { color: theme.ink2, fontSize: 11, fontFamily: theme.font } }) },
      visualMap: { min: Math.min(...values), max: Math.min(1.2, Math.max(...values)), calculable: true, precision: 2, orient: "horizontal", left: "center", bottom: 0,
        text: ["peor", "mejor"], inRange: { color: HEAT_RAMP }, textStyle: { color: theme.ink3, fontFamily: theme.font } },
      series: [{ type: "heatmap", data, itemStyle: { borderColor: theme.surface, borderWidth: 2 }, emphasis: { itemStyle: { borderColor: theme.ink } } }],
    };
  }, [models, rows]);
  return <Chart option={option} height={Math.max(320, models.length * 24 + 90)} label="Mapa de calor del MCRMSE de cada modelo en cada prompt" />;
}

function OptunaChart() {
  const { data } = useApi<PruebaOptuna[]>("/api/optuna");
  const option = useCallback((theme: Theme) => {
    const final = (data ?? []).filter((row) => row.busqueda.startsWith("Final"));
    return {
      grid: { left: 8, right: 16, top: 40, bottom: 36, containLabel: true },
      legend: legend(theme),
      tooltip: tooltip(theme, { trigger: "axis" }),
      xAxis: { type: "value", name: "Prueba", nameLocation: "middle", ...axis(theme) },
      yAxis: { type: "value", name: "RMSE interno", nameLocation: "middle", scale: true, ...axis(theme, { nameGap: 44 }) },
      series: TARGETS.flatMap((target, i) => {
        const rows = final.filter((row) => row.objetivo === target);
        const color = theme.series[i];
        return [
          { name: `${target} · prueba`, type: "scatter", symbol: i === 0 ? "circle" : "triangle", symbolSize: 7, itemStyle: { color, opacity: 0.45 }, data: rows.map((r) => [r.prueba, r.rmse_interno]) },
          { name: `${target} · mejor hasta ahora`, type: "line", step: "end", symbol: "none", lineStyle: { color, width: 2.5 }, itemStyle: { color }, data: rows.map((r) => [r.prueba, r.mejor_hasta_ahora]) },
        ];
      }),
    };
  }, [data]);
  if (!data) return <div className="skeleton chart-tall" role="status" aria-label="Cargando pruebas de Optuna" />;
  return <Chart option={option} height={360} label="RMSE de cada prueba de Optuna y el mejor valor encontrado" />;
}

function ModelCharts({ model, byPrompt }: { model: Modelo; byPrompt: ModeloPorPrompt[] }) {
  const prompts = byPrompt.filter((row) => row.modelo === model.modelo);
  return (
    <div className="chart-grid">
      {model.tiene_predicciones
        ? TARGETS.map((target) => <PredictedVsReal key={target} model={model.modelo} target={target} />)
        : <p className="notice">Este modelo no guardó predicciones por resumen; solo se muestran sus métricas y el error por prompt.</p>}
      {prompts.length > 0 && (
        <figure className="chart-card">
          <figcaption>RMSE por prompt<span>content y wording</span></figcaption>
          <PromptBars rows={prompts} />
        </figure>
      )}
    </div>
  );
}

// En vez de 7,165 puntos encimados: los resúmenes se ordenan por su nota real y se parten en 8 grupos del mismo
// tamaño. Cada caja muestra lo que predijo el modelo para ese grupo, y el rombo, la nota real promedio. Si el modelo
// acertara, cada caja quedaría centrada en su rombo.
const GROUPS = 8;

function PredictedVsReal({ model, target }: { model: string; target: Target }) {
  const { data: real } = useApi<Resumenes>("/api/resumenes");
  const { data: predicted, error } = useApi<{ pred_content: number[]; pred_wording: number[] }>(`/api/predicciones/${encodeURIComponent(model)}`);

  const option = useCallback((theme: Theme) => {
    const y = real![target];
    const x = predicted![target === "content" ? "pred_content" : "pred_wording"];
    const order = y.map((_, i) => i).sort((a, b) => y[a] - y[b]);
    const size = Math.ceil(order.length / GROUPS);
    const groups = Array.from({ length: GROUPS }, (_, g) => order.slice(g * size, (g + 1) * size)).filter((g) => g.length);
    const labels = groups.map((g) => `${fixed(y[g[0]], 1)} a ${fixed(y[g[g.length - 1]], 1)}`);
    const color = theme.series[target === "content" ? 0 : 1];
    return {
      grid: { left: 8, right: 16, top: 40, bottom: 44, containLabel: true },
      legend: legend(theme, { data: ["Predicción del modelo", "Nota real promedio"] }),
      tooltip: tooltip(theme, { trigger: "item", formatter: (p: { seriesName: string; dataIndex: number; data: number[] | number }) => {
        if (p.seriesName === "Nota real promedio") return `Grupo ${labels[p.dataIndex]}<br/>nota real promedio <b>${fixed(p.data as number, 2)}</b>`;
        const [lo, q1, med, q3, hi] = (p.data as number[]).slice(-5);
        return `Grupo ${labels[p.dataIndex]} · ${groups[p.dataIndex].length} resúmenes<br/>predicción mediana <b>${fixed(med, 2)}</b>` +
          `<br/>la mitad entre ${fixed(q1, 2)} y ${fixed(q3, 2)}<br/>90 % entre ${fixed(lo, 2)} y ${fixed(hi, 2)}`;
      } }),
      xAxis: { type: "category", data: labels, name: "Nota real del grupo", nameLocation: "middle",
        ...axis(theme, { nameGap: 44, axisLabel: { color: theme.ink3, fontSize: 11, fontFamily: theme.font, interval: 0, rotate: 30 } }) },
      yAxis: { type: "value", name: "Nota", nameLocation: "middle", scale: true, ...axis(theme, { nameGap: 36 }) },
      series: [
        { name: "Predicción del modelo", type: "boxplot", boxWidth: ["40%", "70%"],
          itemStyle: { color: `${color}33`, borderColor: color, borderWidth: 2 },
          data: groups.map((g) => boxStats(g.map((i) => x[i]))) },
        { name: "Nota real promedio", type: "scatter", symbol: "diamond", symbolSize: 13, z: 5,
          itemStyle: { color: theme.ink, borderColor: theme.surface, borderWidth: 1.5 },
          data: groups.map((g) => mean(g.map((i) => y[i]))) },
      ],
    };
  }, [real, predicted, target]);

  return (
    <figure className="chart-card">
      <figcaption>Predicho vs real · {target}
        <span>{real && predicted ? `RMSE ${fixed(rmse(real[target], predicted[target === "content" ? "pred_content" : "pred_wording"]))}` : "cargando…"}</span>
      </figcaption>
      {error ? <p className="muted">{error}</p> : real && predicted
        ? <Chart option={option} height={340} label={`Cajas con la nota de ${target} predicha en 8 grupos de nota real; si el modelo acierta, cada caja queda centrada en el rombo de la nota real`} />
        : <div className="skeleton chart" role="status" aria-label="Cargando predicciones" />}
      <p className="muted hint flush">Cada caja agrupa la octava parte de los resúmenes, ordenados por su
        nota real. Si la caja queda por encima del rombo, el modelo sobreestima a ese grupo; si queda por debajo, lo subestima.</p>
    </figure>
  );
}

function PromptBars({ rows }: { rows: ModeloPorPrompt[] }) {
  const option = useCallback((theme: Theme) => ({
    grid: { left: 8, right: 16, top: 40, bottom: 24, containLabel: true },
    legend: legend(theme),
    tooltip: tooltip(theme, { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => v.toFixed(3) }),
    xAxis: { type: "category", data: rows.map((r) => short(r.prompt_title, 16)), ...axis(theme) },
    yAxis: { type: "value", min: 0, ...axis(theme) },
    series: TARGETS.map((target, i) => ({
      name: target, type: "bar", barGap: "10%",
      itemStyle: { color: theme.series[i], borderRadius: [4, 4, 0, 0] },
      data: rows.map((r) => (target === "content" ? r.rmse_content : r.rmse_wording)),
    })),
  }), [rows]);
  return <Chart option={option} height={300} label="RMSE de content y wording en cada prompt" />;
}
