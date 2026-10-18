// Inicio: calificar un resumen o varios archivos de resúmenes con el modelo LightGBM.
import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { type Calificacion, type Meta, type Prompt, postJson, type RespuestaLote, type ScoreLote, TARGETS, useApi } from "../api";
import { Dropzone, WarnIcon } from "../components/Dropzone";
import { download, toCsv } from "../csv";
import { usePointer, useScrollProgress } from "../scroll";
import { signed } from "../stats";
import { PaperScene } from "../three/PaperScene";
import { Stage } from "../three/Stage";

const MAX_SUMMARY = 20_000;

export function Home() {
  const hero = useRef<HTMLElement>(null);
  // Las hojas se apilan mientras el encabezado sale de la pantalla
  const progress = useScrollProgress(hero, { start: "top top", end: "bottom 10%" });
  const pointer = usePointer();

  return (
    <>
      <header ref={hero} className="hero rise">
        <Stage className="hero-stage" camera={{ position: [0, 0.4, 5.2], fov: 38 }}
          label="Hojas de papel que flotan y se apilan al bajar la página">
          <PaperScene progress={progress} pointer={pointer} />
        </Stage>
        <h1>Premerie</h1>
        <p>Escribe o sube resúmenes de estudiantes y el modelo LightGBM predice su nota de <strong>content</strong> y
          de <strong>wording</strong>.</p>
      </header>
      <SingleGrade />
      <BatchGrade />
    </>
  );
}

function SingleGrade() {
  const { data: prompts, error: promptsError } = useApi<Prompt[]>("/api/prompts");
  const { data: meta } = useApi<Meta>("/api/meta");
  const [promptId, setPromptId] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<Calificacion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const selected = promptId || prompts?.[0]?.prompt_id || "";
  const over = text.length > MAX_SUMMARY;

  const grade = async () => {
    if (!text.trim() || over || !selected) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await postJson<Calificacion>("/api/calificar", { text, prompt_id: selected }));
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card" aria-labelledby="single-h">
      <h2 id="single-h">Calificar un resumen</h2>
      <div className="field">
        <label htmlFor="prompt-sel">Prompt</label>
        <select id="prompt-sel" className="select" value={selected} onChange={(e) => { setPromptId(e.target.value); setResult(null); }}
          disabled={!prompts}>
          {!prompts && <option>Cargando prompts…</option>}
          {prompts?.map((prompt) => (
            <option key={prompt.prompt_id} value={prompt.prompt_id}>
              {prompt.prompt_title}{prompt.origen === "usuario" ? " (guardado)" : ""}
            </option>
          ))}
        </select>
        {promptsError && <p className="alert" role="alert"><WarnIcon />No se pudo conectar con la API: {promptsError}</p>}
        <span className="muted hint">¿Tu texto no está? <Link to="/agregar">Agrégalo como prompt</Link>.</span>
      </div>

      <div className="field">
        <div className="row-between">
          <label htmlFor="summary">Resumen</label>
          <span className={`counter${over ? " over" : ""}`}>{text.length.toLocaleString("es")} / 20 000 caracteres</span>
        </div>
        <textarea id="summary" rows={7} className={`textarea${over ? " invalid" : ""}`} value={text}
          placeholder="Escribe aquí el resumen del estudiante…" aria-invalid={over}
          onChange={(e) => { setText(e.target.value); setResult(null); }} />
        {over && (
          <p className="alert" role="alert"><WarnIcon />El resumen supera el máximo de 20 000 caracteres. Recórtalo antes de calificar.</p>
        )}
        {meta && <span className="muted hint">
          El modelo se entrenó con resúmenes de {meta.rango_palabras[0]} a {meta.rango_palabras[1]} palabras.</span>}
      </div>

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={grade} disabled={loading || !text.trim() || over || !selected}>
          {loading ? "Calificando…" : "Calificar"}
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M3 9h12M10 4l5 5-5 5" /></svg>
        </button>
      </div>

      {error && <p className="alert" role="alert"><WarnIcon />{error}</p>}
      {result && meta && <GradeResult result={result} meta={meta} />}
    </section>
  );
}

function GradeResult({ result, meta }: { result: Calificacion; meta: Meta }) {
  const names = Object.fromEntries(meta.features.map((feature) => [feature.id, feature.nombre]));
  const maxAbs = Math.max(...TARGETS.flatMap((target) => Object.values(result[target].contribuciones).map(Math.abs)), 0.01);
  const anyCopied = result.tramos.some((tramo) => tramo.copiado);

  return (
    <div className="rise grade-result" aria-live="polite">
      {result.advertencia && <p className="notice">{result.advertencia}</p>}
      <div className="scores">
        {TARGETS.map((target) => (
          <div key={target} className={`score ${target}`}>
            <span className="kicker">{target}</span>
            <span className="value">{signed(result[target].nota)}</span>
            <span className="score-sub">Mejor que el {Math.round(result[target].percentil)} % de los{" "}
              {result.referencia.resumenes.toLocaleString("es")} resúmenes de referencia</span>
          </div>
        ))}
      </div>

      {TARGETS.map((target) => {
        const rows = Object.entries(result[target].contribuciones).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        return (
          <details key={target} className="details">
            <summary>Qué subió o bajó la nota de {target}</summary>
            <div className="contrib">
              {rows.map(([id, value]) => {
                const width = (Math.abs(value) / maxAbs) * 50;
                return (
                  <div key={id} className="contrib-row">
                    <span>{names[id] ?? id}</span>
                    <span className="contrib-track" aria-hidden="true">
                      <span className="contrib-bar" style={{ width: `${width}%`, left: value >= 0 ? "50%" : `${50 - width}%`,
                        background: value >= 0 ? "var(--series-1)" : "var(--series-2)" }} />
                    </span>
                    <span className="contrib-value">{signed(value)}</span>
                  </div>
                );
              })}
            </div>
          </details>
        );
      })}

      {anyCopied && (
        <details className="details">
          <summary>Frases copiadas del texto fuente</summary>
          <p className="source-text">
            {result.tramos.map((tramo, i) => tramo.copiado ? <mark key={i} className="copied">{tramo.texto}</mark> : <span key={i}>{tramo.texto}</span>)}
          </p>
        </details>
      )}
    </div>
  );
}

interface BatchRow extends ScoreLote {
  archivo: string;
}

function BatchGrade() {
  const [files, setFiles] = useState<File[]>([]);
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setErrors([]);
    const collected: BatchRow[] = [];
    const failed: string[] = [];
    for (const file of files) {
      const body = new FormData();
      body.append("resumenes", file);
      try {
        const response = await fetch("/api/calificar/archivo", { method: "POST", body });
        const json = await response.json();
        if (!response.ok) throw new Error(typeof json.detail === "string" ? json.detail : `Error ${response.status}`);
        collected.push(...(json as RespuestaLote).resumenes.map((row) => ({ archivo: file.name, ...row })));
      } catch (e) {
        failed.push(`${file.name}: ${(e as Error).message}`);
      }
    }
    setRows(collected);
    setErrors(failed);
    setLoading(false);
  };

  const shown = useMemo(() => rows.slice(0, 200), [rows]);

  return (
    <section className="card" aria-labelledby="batch-h">
      <div className="section-head">
        <h2 id="batch-h">Calificar en batch</h2>
        <p className="muted flush">Sube uno o varios archivos CSV con las columnas <code>id</code>,{" "}
          <code>prompt_id</code> y <code>text</code> (hasta 5 000 resúmenes por archivo).</p>
      </div>
      <Dropzone files={files} onChange={(next) => { setFiles(next); setRows([]); setErrors([]); }}
        title="Elegir archivos de tu computadora" hint="o arrástralos aquí · .csv" />
      <div className="row-between middle">
        <a href="/api/plantillas/resumenes.csv" download>Descargar plantilla CSV</a>
        <button type="button" className="btn btn-ghost" onClick={run} disabled={!files.length || loading}>
          {loading ? "Calificando…" : `Calificar ${files.length > 1 ? `${files.length} archivos` : "archivo"}`}
        </button>
      </div>
      {errors.map((message) => <p key={message} className="alert" role="alert"><WarnIcon />{message}</p>)}
      {rows.length > 0 && (
        <div className="rise batch-results">
          <div className="row-between middle">
            <strong>{rows.length.toLocaleString("es")} resúmenes calificados</strong>
            <button type="button" className="btn btn-primary" onClick={() => download("resumenes_calificados.csv", toCsv(rows as unknown as Record<string, unknown>[]))}>
              Descargar resultados
            </button>
          </div>
          <div className="table-box">
            <table className="table">
              <thead>
                <tr><th>Archivo</th><th>id</th><th>Prompt</th><th className="num">Content</th><th className="num">Percentil</th>
                  <th className="num">Wording</th><th className="num">Percentil</th><th>Aviso</th></tr>
              </thead>
              <tbody>
                {shown.map((row) => (
                  <tr key={`${row.archivo}-${row.id}`}>
                    <td>{row.archivo}</td><td>{row.id}</td><td>{row.prompt_id}</td>
                    <td className="num">{signed(row.content)}</td><td className="num">{Math.round(row.percentil_content)}</td>
                    <td className="num">{signed(row.wording)}</td><td className="num">{Math.round(row.percentil_wording)}</td>
                    <td className="muted hint">{row.advertencia ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows.length > shown.length && <span className="muted hint">Se muestran los primeros 200; el archivo descargado trae todos.</span>}
        </div>
      )}
    </section>
  );
}
