// Zona para elegir varios archivos del sistema de archivos (o arrastrarlos) y su lista.
import { type DragEvent, useId, useState } from "react";

interface Props {
  files: File[];
  onChange: (files: File[]) => void;
  title: string;
  hint: string;
  accept?: string;
  compact?: boolean;
}

const size = (bytes: number) => (bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`);

function UploadIcon({ px }: { px: number }) {
  return (
    <svg width={px} height={px} viewBox="0 0 36 36" fill="none" stroke="#5A3414" strokeWidth="1.8" strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true"><path d="M18 24V8M11 15l7-7 7 7" /><path d="M6 24v4h24v-4" /></svg>
  );
}

export function Dropzone({ files, onChange, title, hint, accept = ".csv,text/csv", compact = false }: Props) {
  const id = useId();
  const [dragging, setDragging] = useState(false);

  // Agrega sin repetir archivos con el mismo nombre y tamaño
  const add = (incoming: File[]) => {
    const key = (file: File) => `${file.name}-${file.size}`;
    const known = new Set(files.map(key));
    onChange([...files, ...incoming.filter((file) => !known.has(key(file)))]);
  };

  const drop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    add(Array.from(event.dataTransfer.files).filter((file) => file.name.toLowerCase().endsWith(".csv")));
  };

  return (
    <div className="dropzone-wrap">
      <label htmlFor={id} className={`dropzone${compact ? " compact" : ""}${dragging ? " dragging" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)} onDrop={drop}>
        <UploadIcon px={compact ? 26 : 36} />
        <span className="dropzone-text">
          <strong>{title}</strong>
          <span className="muted hint">{hint}</span>
        </span>
        <input id={id} type="file" multiple accept={accept}
          onChange={(event) => { add(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
      </label>
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((file, i) => (
            <li key={`${file.name}-${i}`}>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M4 2h7l3 3v11H4z" /></svg>
              <span className="name">{file.name}</span>
              <span className="muted mono">{size(file.size)}</span>
              <button type="button" className="icon-btn" aria-label={`Quitar ${file.name}`}
                onClick={() => onChange(files.filter((_, j) => j !== i))}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function WarnIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
      aria-hidden="true" className="no-shrink"><path d="M9 2l7.5 13h-15z" /><path d="M9 7v4M9 13.5v.1" /></svg>
  );
}
