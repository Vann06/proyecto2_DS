// Agregar prompts (textos fuente): uno con el formulario o varios desde archivos CSV.
import { useState } from "react";

import { invalidate, type Prompt, postJson, useApi } from "../api";
import { Dropzone, WarnIcon } from "../components/Dropzone";
import { parseCsv } from "../csv";

const MAX_TITLE = 200;
const MAX_QUESTION = 1_000;
const MAX_TEXT = 50_000;

interface Saved {
  prompt_id: string;
  prompt_title: string;
}

async function savePrompt(body: Record<string, string>) {
  const saved = await postJson<Saved>("/api/prompts", body);
  invalidate("/api/prompts");
  return saved;
}

export function AddPrompt() {
  const [title, setTitle] = useState("");
  const [question, setQuestion] = useState("");
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [messages, setMessages] = useState<{ ok: boolean; text: string }[]>([]);
  const [sending, setSending] = useState(false);

  const over = text.length > MAX_TEXT;
  const single = title.trim() && text.trim() && !over;

  const send = async () => {
    setSending(true);
    const log: { ok: boolean; text: string }[] = [];
    if (single) {
      try {
        const saved = await savePrompt({ prompt_title: title, prompt_question: question, prompt_text: text });
        log.push({ ok: true, text: `Se guardó «${saved.prompt_title}» (id ${saved.prompt_id}).` });
        setTitle(""); setQuestion(""); setText("");
      } catch (e) {
        log.push({ ok: false, text: `«${title}»: ${(e as Error).message}` });
      }
    }
    for (const file of files) {
      const rows = parseCsv(await file.text());
      if (!rows.length || !("prompt_text" in rows[0])) {
        log.push({ ok: false, text: `${file.name}: el archivo debe tener las columnas prompt_title y prompt_text.` });
        continue;
      }
      let ok = 0;
      for (const [i, row] of rows.entries()) {
        try {
          await savePrompt({
            prompt_title: row.prompt_title || row.prompt_id || `Prompt ${i + 1}`,
            prompt_text: row.prompt_text,
            prompt_question: row.prompt_question ?? "",
            ...(row.prompt_id ? { prompt_id: row.prompt_id } : {}),
          });
          ok++;
        } catch (e) {
          log.push({ ok: false, text: `${file.name}, fila ${i + 2}: ${(e as Error).message}` });
        }
      }
      if (ok) log.push({ ok: true, text: `${file.name}: se guardaron ${ok} de ${rows.length} prompts.` });
    }
    setFiles([]);
    setMessages(log);
    setSending(false);
  };

  return (
    <>
      <header className="page-head center rise">
        <h1 className="page-title">Agregar un nuevo prompt</h1>
        <p>Un prompt es el texto fuente que los estudiantes resumen. Guárdalo para poder calificar resúmenes sobre él.</p>
      </header>

      <form className="card" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <div className="form-grid">
          <label htmlFor="p-name" className="label">Nombre del prompt</label>
          <input id="p-name" className="input" maxLength={MAX_TITLE} value={title} placeholder="Ej. On Tragedy"
            onChange={(e) => setTitle(e.target.value)} />

          <label htmlFor="p-question" className="label">Pregunta <span className="muted optional">(opcional)</span></label>
          <input id="p-question" className="input" maxLength={MAX_QUESTION} value={question}
            placeholder="Ej. Summarize at least 3 elements of an ideal tragedy…" onChange={(e) => setQuestion(e.target.value)} />

          <label htmlFor="p-text" className="label top">Prompt</label>
          <div className="field">
            <textarea id="p-text" rows={8} className={`textarea${over ? " invalid" : ""}`} value={text} aria-invalid={over}
              placeholder="Pega aquí el texto fuente completo…" onChange={(e) => setText(e.target.value)} />
            <span className={`counter${over ? " over" : ""}`}>
              {text.length.toLocaleString("es")} / 50 000 caracteres máx.</span>
            {over && <p className="alert" role="alert"><WarnIcon />El texto supera los 50 000 caracteres permitidos.</p>}
          </div>
        </div>

        <div className="divider">o agrega varios</div>

        <div className="form-grid">
          <span className="label top">Agregar en batch</span>
          <div className="field">
            <Dropzone compact files={files} onChange={setFiles} title="Subir archivos"
              hint="CSV con prompt_title, prompt_text y prompt_question (opcional)" />
            <a href="/api/plantillas/prompts.csv" download className="small">Descargar plantilla CSV</a>
          </div>
        </div>

        {messages.map((message, i) => message.ok
          ? <p key={i} className="notice">{message.text}</p>
          : <p key={i} className="alert" role="alert"><WarnIcon />{message.text}</p>)}

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={sending || (!single && !files.length)}>
            {sending ? "Enviando…" : "Enviar"}
          </button>
        </div>
      </form>

      <SavedPrompts />
    </>
  );
}

function SavedPrompts() {
  const { data } = useApi<Prompt[]>("/api/prompts");
  const saved = data?.filter((prompt) => prompt.origen === "usuario") ?? [];
  const [error, setError] = useState<string | null>(null);
  // Borrar no se puede deshacer: el basurero primero pide confirmar en la misma fila
  const [confirming, setConfirming] = useState<string | null>(null);
  if (!saved.length) return null;

  const remove = async (id: string) => {
    setConfirming(null);
    const response = await fetch(`/api/prompts/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!response.ok) setError(`No se pudo borrar «${id}».`);
    invalidate("/api/prompts");
  };

  return (
    <section className="card" aria-labelledby="saved-h">
      <h2 id="saved-h">Prompts guardados</h2>
      {error && <p className="alert" role="alert"><WarnIcon />{error}</p>}
      <ul className="file-list">
        {saved.map((prompt) => (
          <li key={prompt.prompt_id}>
            <span className="name"><strong>{prompt.prompt_title}</strong> <span className="muted">· {prompt.prompt_id}</span></span>
            {confirming === prompt.prompt_id ? (
              <span className="confirm" role="group" aria-label={`Confirmar borrado de ${prompt.prompt_title}`}>
                ¿Borrar?
                <button type="button" className="link-btn" onClick={() => remove(prompt.prompt_id)}>Borrar</button>
                <button type="button" className="link-btn" onClick={() => setConfirming(null)} autoFocus>Cancelar</button>
              </span>
            ) : (
              <button type="button" className="icon-btn" aria-label={`Borrar ${prompt.prompt_title}`} onClick={() => setConfirming(prompt.prompt_id)}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true"><path d="M3 4h10M6 4V2.5h4V4M5 4l.6 9h4.8L11 4" /></svg>
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
