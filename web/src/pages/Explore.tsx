// Cómo funciona: el camino de un resumen hasta sus notas, contado con scroll.
import { Link } from "react-router-dom";

import { type Meta, type Modelo, type Prompt, useApi } from "../api";
import { Diagram, type DiagramEdge, type DiagramNode } from "../components/Diagram";
import { Reveal } from "../components/Reveal";
import { MachineScene } from "../three/MachineScene";
import { Stage } from "../three/Stage";
import { fixed } from "../stats";

const TIPS = {
  prompt: "Texto fuente (el prompt): lo que leyó el estudiante.",
  resumen: "Resumen: lo que escribió el estudiante; es lo que se califica.",
  tokPrompt: "Tokenizar: partir el texto en palabras en minúsculas con la misma regla que el resumen, para poder compararlos.",
  tokResumen: "Tokenizar: la misma regla aplicada al resumen; de aquí salen el conteo de palabras y los n-gramas.",
  features: "7 features manuales: las 4 de largo y vocabulario y las 3 de palabras compartidas con el texto fuente.",
};

// Las 7 features son de dos tipos: 4 salen solo del resumen y 3 lo comparan con el texto fuente (src/features.py)
const STEP1_NODES: DiagramNode[] = [
  { id: "prompt", label: "Texto fuente", x: 80, y: 30, at: 0, tip: TIPS.prompt },
  { id: "resumen", label: "Resumen", x: 620, y: 30, at: 0.05, tip: TIPS.resumen },
  { id: "tok1", label: "Tokenizar", x: 80, y: 150, at: 0.2, tip: TIPS.tokPrompt },
  { id: "tok2", label: "Tokenizar", x: 620, y: 150, at: 0.24, tip: TIPS.tokResumen },
  { id: "shared", label: "Palabras compartidas · 3", x: 220, y: 280, w: 300, at: 0.45, tip: "Solo con el texto fuente al lado: qué parte de las palabras, de los pares y de los tríos de palabras distintos del resumen aparece también en el texto. Mide cuánto se copió." },
  { id: "vocab", label: "Largo y vocabulario · 4", x: 640, y: 280, w: 300, at: 0.45, tip: "Solo dependen del resumen: número de palabras, su logaritmo, palabras distintas y diversidad léxica (distintas entre totales)." },
  { id: "feat", label: "7 features manuales", x: 360, y: 410, w: 280, accent: true, at: 0.7, tip: TIPS.features },
];
const STEP1_EDGES: DiagramEdge[] = [
  { d: "M190 86V146", at: 0.08 },
  { d: "M730 86V146", at: 0.12 },
  { d: "M190 206C190 248 300 250 330 276", at: 0.32 },
  { d: "M680 206C680 250 460 250 430 276", at: 0.36 },
  { d: "M790 206V276", at: 0.36 },
  { d: "M370 336C370 380 430 384 450 406", at: 0.58 },
  { d: "M790 336C790 380 570 384 550 406", at: 0.6 },
];

const STEP2_NODES: DiagramNode[] = [
  { id: "prompt", label: "Texto fuente", x: 80, y: 30, at: 0, tip: TIPS.prompt },
  { id: "resumen", label: "Resumen", x: 620, y: 30, at: 0.05, tip: TIPS.resumen },
  { id: "emb1", label: "Embedding spaCy", x: 80, y: 170, at: 0.28, tip: "Embedding de spaCy (en_core_web_md): el promedio de los vectores de 300 dimensiones de las palabras del prompt." },
  { id: "emb2", label: "Embedding spaCy", x: 620, y: 170, at: 0.32, tip: "Embedding de spaCy del resumen, calculado de la misma forma." },
  { id: "cos", label: "cos_sim", x: 350, y: 300, w: 300, accent: true, at: 0.68, tip: "cos_sim: el coseno entre los dos vectores. Cerca de 1 quiere decir que hablan del mismo tema." },
];
const STEP2_EDGES: DiagramEdge[] = [
  { d: "M190 86V166", at: 0.1 },
  { d: "M730 86V166", at: 0.15 },
  { d: "M190 226C190 296 290 328 346 328", at: 0.45 },
  { d: "M730 226C730 296 710 328 654 328", at: 0.48 },
];

const STEP3_NODES: DiagramNode[] = [
  { id: "cos", label: "cos_sim", x: 140, y: 20, at: 0, tip: "cos_sim entra como la octava columna." },
  { id: "feat", label: "7 features", x: 640, y: 20, at: 0.03, tip: "Las 7 features manuales calculadas en el paso 1." },
  { id: "fusion", label: "Fusión temprana", x: 390, y: 130, accent: true, at: 0.2, tip: "Fusión temprana: las 8 columnas se juntan en una sola tabla antes de entrenar." },
  // La máquina es LightGBM: la placa va pegada debajo de ella, y de la máquina salen las dos notas
  { id: "lgbm", label: "LightGBM", x: 400, y: 436, w: 200, plate: true, at: 0.36, tip: "LightGBM es la máquina: adentro hay dos modelos de árboles, uno que predice content y otro wording. Aprendieron de los 7,165 resúmenes con sus notas reales, con hiperparámetros elegidos con Optuna (notebook 11)." },
  { id: "content", label: "Nota de content", x: 120, y: 540, w: 260, accent: true, at: 0.75, tip: "Content: nota predicha de ideas principales y detalle." },
  { id: "wording", label: "Nota de wording", x: 620, y: 540, w: 260, accent: true, at: 0.8, tip: "Wording: nota predicha de lenguaje propio y claro." },
];
const STEP3_EDGES: DiagramEdge[] = [
  { d: "M250 76C250 110 400 110 440 128", at: 0.06 },
  { d: "M750 76C750 110 600 110 560 128", at: 0.08 },
  { d: "M500 186V214", at: 0.3 },
  { d: "M430 480C340 492 260 500 250 536", at: 0.58 },
  { d: "M570 480C660 492 740 500 750 536", at: 0.62 },
];

export function Explore() {
  const { data: meta } = useApi<Meta>("/api/meta");
  const { data: prompts } = useApi<Prompt[]>("/api/prompts");
  const tragedy = prompts?.find((prompt) => prompt.prompt_id === "39c16e");
  const excerpt = tragedy ? tragedy.prompt_text.replace(/\s+/g, " ").slice(0, 120).trim() + "…" : "…";
  const { data: models } = useApi<{ modelos: Modelo[] }>("/api/modelos");
  const app = models?.modelos.find((model) => model.es_modelo_aplicacion);

  return (
    <>
      <section className="story intro rise">
        <h1>¿Cómo funciona<br />nuestro modelo?</h1>
        <p>Baja para seguir el camino de un resumen: desde el texto que leyó el estudiante hasta las dos notas que predice LightGBM.</p>
        <span className="muted scroll-hint">
          <svg width="20" height="28" viewBox="0 0 20 28" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="2" y="2" width="16" height="24" rx="8" /><path d="M10 7v5" strokeLinecap="round" /></svg>
          Scroll
        </span>
      </section>

      <section className="story" aria-labelledby="prompt-h">
        <h2 id="prompt-h">¿Qué es un prompt?</h2>
        <div className="def-list">
          <Reveal className="def-row">
            <div className="term"><small>Nombre del prompt</small><span>{tragedy?.prompt_title ?? "On Tragedy"}</span></div>
            <p>Identifica de qué texto se habla. Los {meta ? meta.resumenes.toLocaleString("es") : "7 165"} resúmenes de
              entrenamiento vienen de 4 prompts de CommonLit, y la validación deja uno fuera cada vez.</p>
          </Reveal>
          <Reveal className="def-row">
            <div className="term"><small>Prompt (texto fuente)</small><span className="quote">“{excerpt}”</span></div>
            <p>El texto que leyó el estudiante. Con él se miden el traslape de palabras y frases y la similitud de tema: es
              la base de las features manuales.</p>
          </Reveal>
        </div>
      </section>

      <section className="band-sand" aria-labelledby="sum-h">
        <div className="story">
          <h2 id="sum-h">¿Qué es un resumen?</h2>
          <div className="def-list">
            <Reveal className="def-row">
              <div className="term"><small>Resumen (ejemplo)</small>
                <span className="quote">“Aristotle says an ideal tragedy makes the audience feel pity and fear, and the hero falls because of a mistake…”</span></div>
              <p>Lo que escribió el estudiante después de leer el prompt. Es la entrada que el modelo califica.</p>
            </Reveal>
          </div>
          <h3 className="subhead">¿Qué queremos predecir?</h3>
          <div className="target-grid">
            <Reveal className="target">
              <b>Content</b>
              <p>Qué tan bien captura las ideas principales del texto fuente y si las explica con detalle y sin errores.</p>
            </Reveal>
            <Reveal className="target dark" delay={0.25}>
              <b>Wording</b>
              <p>Qué tan propio y claro es el lenguaje: vocabulario, estructura de las oraciones y cuánto evita copiar el texto fuente.</p>
            </Reveal>
          </div>
        </div>
      </section>

      <Diagram eyebrow="Paso 1" title="Features manuales del resumen"
        nodes={STEP1_NODES} edges={STEP1_EDGES} width={1000} height={476} />

      <Diagram eyebrow="Paso 2" title="Similitud de tema con embeddings" nodes={STEP2_NODES} edges={STEP2_EDGES}
        width={1000} height={380} />

      <Diagram eyebrow="Paso 3" title="Fusión temprana y LightGBM" className="band-clay" nodes={STEP3_NODES}
        edges={STEP3_EDGES} width={1000} height={610}>
        {(progress) => (
          <Stage className="machine-stage" camera={{ position: [0, 0.5, 4.6], fov: 36 }}
            label="Máquina que representa al modelo LightGBM; sus engranes giran al bajar">
            <MachineScene progress={progress} />
          </Stage>
        )}
      </Diagram>

      {/* Lo que la máquina no produce pero sí responde: qué tan cerca quedan sus notas de las reales */}
      <section className="story" aria-labelledby="eval-h">
        <h2 id="eval-h">¿Qué tan bien predice?</h2>
        <div className="def-list">
          <Reveal className="def-row">
            <div className="term"><small>MCRMSE</small><span>{app ? fixed(app.mcrmse) : "…"}</span></div>
            <p>Para medirlo, cada vez se entrenó con 3 de los 4 textos y se calificaron los resúmenes del texto que quedó
              fuera, como si fuera uno nuevo. Las notas predichas se comparan con las reales: el MCRMSE es el promedio
              del error (RMSE) de content y de wording. Menos es mejor; predecir siempre el promedio da 1.050.</p>
          </Reveal>
          <Reveal className="def-row">
            <div className="term"><small>Error típico por nota</small>
              <span>{app ? `content ±${fixed(app.rmse_content, 2)} · wording ±${fixed(app.rmse_wording, 2)}` : "…"}</span></div>
            <p>Wording es más difícil: depende de cómo está escrito el resumen, y las features miden sobre todo largo y
              cuánto se parece al texto fuente.</p>
          </Reveal>
        </div>
        <Link to="/metricas" className="link-btn closing-link">Ver la comparación con los 23 modelos</Link>
      </section>

      <section className="story closing" aria-labelledby="feat-h">
        <h2 id="feat-h">Las 7 features manuales y cos_sim</h2>
        <div className="feature-list">
          {meta?.features.map((feature, i) => (
            <Reveal key={feature.id} className={`feature-row${feature.id === "cos_sim" ? " alt" : ""}`} delay={(i % 3) * 0.1}>
              <div className="feature-name">
                <strong>{feature.nombre}</strong>
                <code>{feature.id}</code>
              </div>
              <p>{feature.detalle}</p>
            </Reveal>
          ))}
          {!meta && [0, 1, 2].map((i) => <div key={i} className="skeleton feature" aria-hidden="true" />)}
        </div>
        <Link to="/" className="btn btn-primary closing-cta">Probar con un resumen</Link>
      </section>
    </>
  );
}
