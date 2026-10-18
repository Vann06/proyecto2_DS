// Tipos y llamadas a la API de FastAPI (api/main.py).
import { useEffect, useState } from "react";

export type Target = "content" | "wording";
export const TARGETS: Target[] = ["content", "wording"];

export interface Feature {
  id: string;
  nombre: string;
  detalle: string;
}

export interface Meta {
  features: Feature[];
  objetivos: Target[];
  modelo_aplicacion: { nombre: string; mcrmse: number };
  mejor_modelo: { nombre: string; mcrmse: number };
  rango_palabras: [number, number];
  resumenes: number;
}

export interface Prompt {
  prompt_id: string;
  prompt_title: string;
  prompt_question: string;
  prompt_text: string;
  origen: "commonlit" | "usuario";
  creado: string | null;
  // Solo los 4 textos de CommonLit tienen resúmenes de entrenamiento
  resumenes: number | null;
  content_media: number | null;
  wording_media: number | null;
}

export interface ScoreLote {
  id: string;
  prompt_id: string;
  content: number;
  wording: number;
  percentil_content: number;
  percentil_wording: number;
  advertencia: string | null;
}

export interface RespuestaLote {
  resumenes: ScoreLote[];
  resumen_por_prompt: { prompt_id: string; resumenes: number; content_media: number; wording_media: number;
    con_advertencia: number }[];
  prompts_guardados?: string[];
}

// Una columna por campo, todas del mismo largo (7,165 resúmenes)
export type Resumenes = { prompt_id: string[]; prompt_title: string[] } & Record<Target, number[]> &
  Record<string, (number | string)[]>;

export interface Modelo {
  modelo: string;
  familia: string;
  notebook: string;
  etapa: "final" | "antes del ajuste" | "variante descartada";
  rmse_content: number;
  r2_content: number;
  rmse_wording: number;
  r2_wording: number;
  mcrmse: number;
  desviacion_entre_prompts: number;
  tiene_predicciones: boolean;
  es_modelo_aplicacion: boolean;
  es_mejor_modelo: boolean;
}

export interface ModeloPorPrompt {
  modelo: string;
  prompt_id: string;
  prompt_title: string;
  rmse_content: number;
  rmse_wording: number;
  mcrmse: number;
}

export interface PruebaOptuna {
  prueba: number;
  rmse_interno: number;
  mejor_hasta_ahora: number;
  busqueda: string;
  objetivo: Target;
  segundos: number;
  [param: string]: number | string;
}

export interface Calificacion {
  features: Record<string, number>;
  advertencia: string | null;
  content: Nota;
  wording: Nota;
  referencia: { resumenes: number; prompt_id: string | null; content: number[]; wording: number[] };
  rangos_tipicos: Record<string, { p10: number; mediana: number; p90: number }>;
  tramos: { texto: string; copiado: boolean }[];
}

export interface Nota {
  nota: number;
  base: number;
  percentil: number;
  contribuciones: Record<string, number>;
}

export interface LabParams {
  n_estimators: number;
  learning_rate: number;
  num_leaves: number;
  min_child_samples: number;
  subsample: number;
  colsample_bytree: number;
  reg_lambda: number;
  reg_alpha: number;
}

export interface LabResult {
  rmse: number;
  por_prompt: { prompt_id: string; prompt_title: string; rmse: number }[];
}

export interface LabReferencias {
  parametros_ajustados: LabParams;
  ajustados: LabResult;
  parametros_08: LabResult;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join(" ") : detail;
    throw new Error(message || `La API respondió con el error ${response.status}.`);
  }
  return response.json();
}

export function postJson<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

// Las respuestas GET se guardan para no volver a pedirlas al cambiar de página
const cache = new Map<string, Promise<unknown>>();

export function getJson<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    cache.set(url, request<T>(url).catch((error) => {
      cache.delete(url);
      throw error;
    }));
  }
  return cache.get(url) as Promise<T>;
}

// Avisa a los componentes que usan una URL que deben volver a pedirla (por ejemplo, /api/prompts tras guardar uno)
const listeners = new Map<string, Set<() => void>>();

export function invalidate(url: string) {
  cache.delete(url);
  listeners.get(url)?.forEach((listener) => listener());
}

export function useApi<T>(url: string | null): { data: T | null; error: string | null } {
  const [state, setState] = useState<{ url: string | null; data: T | null; error: string | null }>({
    url: null, data: null, error: null,
  });
  const [version, setVersion] = useState(0);
  useEffect(() => {
    if (!url) return;
    const listener = () => setVersion((value) => value + 1);
    if (!listeners.has(url)) listeners.set(url, new Set());
    listeners.get(url)!.add(listener);
    return () => {
      listeners.get(url)?.delete(listener);
    };
  }, [url]);
  useEffect(() => {
    if (!url) return;
    let active = true;
    getJson<T>(url).then(
      (data) => active && setState({ url, data, error: null }),
      (error: Error) => active && setState({ url, data: null, error: error.message }),
    );
    return () => {
      active = false;
    };
  }, [url, version]);
  return state.url === url ? { data: state.data, error: state.error } : { data: null, error: null };
}
