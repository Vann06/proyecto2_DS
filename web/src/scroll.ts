// Animación ligada al scroll: GSAP ScrollTrigger da el progreso crudo de cada sección y un único
// requestAnimationFrame lo suaviza con lerp, para que el movimiento siga al scroll sin saltos.
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { type RefObject, useEffect, useRef } from "react";

gsap.registerPlugin(ScrollTrigger);

export const lerp = (from: number, to: number, t: number) => from + (to - from) * t;
export const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

export const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export interface Smoothed {
  target: number;
  value: number;
}

interface Entry {
  smoothed: Smoothed;
  ease: number;
  onChange?: (value: number) => void;
}

// Un solo bucle para todos los valores suavizados; se detiene cuando todos llegaron a su objetivo
const entries = new Set<Entry>();
let frame = 0;
let last = 0;

function tick(now: number) {
  const dt = last ? Math.min(0.1, (now - last) / 1000) : 1 / 60;
  last = now;
  let moving = false;
  for (const entry of entries) {
    const { smoothed } = entry;
    const delta = smoothed.target - smoothed.value;
    if (Math.abs(delta) < 0.0005) {
      if (smoothed.value !== smoothed.target) {
        smoothed.value = smoothed.target;
        entry.onChange?.(smoothed.value);
      }
      continue;
    }
    // El factor se ajusta al tiempo entre cuadros: se ve igual a 60 Hz que a 144 Hz
    smoothed.value = lerp(smoothed.value, smoothed.target, 1 - Math.pow(1 - entry.ease, dt * 60));
    entry.onChange?.(smoothed.value);
    moving = true;
  }
  frame = moving ? requestAnimationFrame(tick) : 0;
  if (!moving) last = 0;
}

function wake() {
  if (!frame) frame = requestAnimationFrame(tick);
}

export function setTarget(smoothed: Smoothed, target: number) {
  smoothed.target = target;
  wake();
}

interface Options {
  start?: string;
  end?: string;
  pin?: boolean;
  ease?: number;
  onChange?: (value: number) => void;
}

/** Progreso (0 a 1) de una sección según el scroll, suavizado con lerp. Devuelve un ref para leerlo sin
 * volver a renderizar (en useFrame de R3F o en `onChange`, que escribe directo en el DOM). */
export function useScrollProgress(ref: RefObject<HTMLElement | null>, options: Options = {}) {
  const smoothed = useRef<Smoothed>({ target: 0, value: 0 });
  const onChange = useRef(options.onChange);
  onChange.current = options.onChange;
  const { start = "top 80%", end = "bottom 30%", pin = false, ease = 0.08 } = options;

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const entry: Entry = { smoothed: smoothed.current, ease, onChange: (value) => onChange.current?.(value) };
    if (prefersReducedMotion()) {
      smoothed.current.target = smoothed.current.value = 1;
      entry.onChange?.(1);
      return;
    }
    entries.add(entry);
    const context = gsap.context(() => {
      ScrollTrigger.create({
        trigger: element,
        start,
        end,
        pin,
        anticipatePin: pin ? 1 : 0,
        onUpdate: (self) => setTarget(smoothed.current, self.progress),
        onRefresh: (self) => setTarget(smoothed.current, self.progress),
      });
    });
    entry.onChange?.(smoothed.current.value);
    return () => {
      context.revert();
      entries.delete(entry);
    };
  }, [ref, start, end, pin, ease]);

  return smoothed;
}

/** Valor suavizado que sigue al puntero (−1 a 1 en cada eje), para el paralaje de las escenas 3D. */
export function usePointer() {
  const pointer = useRef({ x: { target: 0, value: 0 }, y: { target: 0, value: 0 } });
  useEffect(() => {
    if (prefersReducedMotion()) return;
    const x: Entry = { smoothed: pointer.current.x, ease: 0.06 };
    const y: Entry = { smoothed: pointer.current.y, ease: 0.06 };
    entries.add(x);
    entries.add(y);
    const move = (event: PointerEvent) => {
      setTarget(pointer.current.x, (event.clientX / window.innerWidth) * 2 - 1);
      setTarget(pointer.current.y, (event.clientY / window.innerHeight) * 2 - 1);
    };
    window.addEventListener("pointermove", move, { passive: true });
    return () => {
      window.removeEventListener("pointermove", move);
      entries.delete(x);
      entries.delete(y);
    };
  }, []);
  return pointer;
}

export { ScrollTrigger };
