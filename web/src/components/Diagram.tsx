// Diagrama de flujo que se dibuja con el scroll: la sección queda fija (pin) mientras las flechas se trazan
// y los pasos aparecen; cada paso explica qué es al pasar el cursor o enfocarlo con el teclado.
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";

import { clamp01, ScrollTrigger, type Smoothed, useScrollProgress } from "../scroll";

export interface DiagramNode {
  id: string;
  label: string;
  x: number;
  y: number;
  w?: number;
  accent?: boolean;
  /** Placa con el nombre de la máquina 3D (tinta oscura, como la máquina) */
  plate?: boolean;
  /** Momento del scroll (0 a 1) en que aparece */
  at: number;
  tip: string;
}

export interface DiagramEdge {
  d: string;
  at: number;
}

interface Props {
  eyebrow: string;
  title: string;
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  width: number;
  height: number;
  className?: string;
  /** Contenido extra dentro del diagrama (la escena 3D), que recibe el progreso suavizado */
  children?: (progress: { current: Smoothed }) => ReactNode;
}

const DRAW = 0.16;
const FADE = 0.1;

export function Diagram({ eyebrow, title, nodes, edges, width, height, className, children }: Props) {
  const section = useRef<HTMLElement>(null);
  const paths = useRef<(SVGPathElement | null)[]>([]);
  const flows = useRef<(SVGPathElement | null)[]>([]);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const lengths = useRef<number[]>([]);
  const [hover, setHover] = useState<DiagramNode | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState({ scale: 1, left: 0 });

  // Mientras la sección está fija, el diagrama completo debe caber en la pantalla: se escala al ancho y al alto
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const update = () => {
      // Alto que queda bajo el título de la sección (se mide, porque el título crece en pantallas grandes) menos
      // el relleno de abajo. El diagrama crece hasta llenar ese espacio y el ancho, con un tope de 1.8.
      const header = section.current ? element.getBoundingClientRect().top - section.current.getBoundingClientRect().top : 200;
      const available = window.innerHeight - header - 48;
      const scale = Math.max(0.42, Math.min(1.8, element.clientWidth / width, available / height));
      setFit({ scale, left: Math.max(0, (element.clientWidth - width * scale) / 2) });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    window.addEventListener("resize", update);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [width, height]);

  // Cambió la altura de la sección fija: las posiciones de los disparadores se recalculan
  useEffect(() => {
    const id = requestAnimationFrame(() => ScrollTrigger.refresh());
    return () => cancelAnimationFrame(id);
  }, [fit.scale]);

  useLayoutEffect(() => {
    lengths.current = paths.current.map((path) => path?.getTotalLength() ?? 0);
    paths.current.forEach((path, i) => {
      if (path) path.style.strokeDasharray = `${lengths.current[i]}`;
    });
  }, [edges]);

  // Escribe directo en el DOM en cada cuadro del lerp, sin volver a renderizar React
  const apply = (p: number) => {
    edges.forEach((edge, i) => {
      const drawn = clamp01((p - edge.at) / DRAW);
      const path = paths.current[i];
      if (path) path.style.strokeDashoffset = `${lengths.current[i] * (1 - drawn)}`;
      const flow = flows.current[i];
      if (flow) flow.style.opacity = drawn >= 1 ? "1" : "0";
    });
    nodes.forEach((node, i) => {
      const shown = clamp01((p - node.at) / FADE);
      const button = buttons.current[i];
      if (!button) return;
      button.style.opacity = `${shown}`;
      button.style.transform = `translateY(${(1 - shown) * 14}px) scale(${0.92 + 0.08 * shown})`;
    });
  };

  // La sección ocupa toda la pantalla y se queda fija durante 1.4 pantallas de scroll: es el recorrido en que se
  // dibujan las flechas y aparecen los pasos
  const progress = useScrollProgress(section, { start: "top top", end: "+=140%", pin: true, onChange: apply });

  // El div exterior es de React y GSAP no lo toca: al fijar la sección, GSAP la mete dentro de un «pin-spacer»
  // propio. Sin este div, al cambiar de página React intenta quitar la sección de un padre que ya no es el suyo,
  // falla (removeChild) y la aplicación entera se queda en blanco.
  return (
    <div>
    <section ref={section} className={`pin-section ${className ?? ""}`} aria-labelledby={`${eyebrow}-h`}>
      <div className="story diagram-story">
        <span className="eyebrow">{eyebrow}</span>
        <h2 id={`${eyebrow}-h`}>{title}</h2>
        <div ref={box} className="diagram-box" style={{ height: height * fit.scale + 16 }}>
          <div className="diagram" style={{ width, height, transform: `translate(${fit.left}px, 8px) scale(${fit.scale})`, transformOrigin: "0 0" }}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
              <defs>
                <marker id={`arrow-${eyebrow}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                  <path d="M0 0L10 5L0 10z" fill="var(--brown)" />
                </marker>
              </defs>
              {edges.map((edge, i) => (
                <g key={i}>
                  <path ref={(el) => { paths.current[i] = el; }} className="edge" d={edge.d} markerEnd={`url(#arrow-${eyebrow})`} />
                  <path ref={(el) => { flows.current[i] = el; }} className="edge-flow" d={edge.d} />
                </g>
              ))}
            </svg>
            {children?.(progress)}
            {nodes.map((node, i) => (
              <button key={node.id} ref={(el) => { buttons.current[i] = el; }} type="button"
                className={`node${node.accent ? " accent" : ""}${node.plate ? " plate" : ""}${hover?.id === node.id ? " on" : ""}`}
                style={{ left: node.x, top: node.y, width: node.w ?? 220 }}
                onMouseEnter={() => setHover(node)} onFocus={() => setHover(node)}
                onMouseLeave={() => setHover(null)} onBlur={() => setHover(null)}>
                {node.label}
                <span className="sr-only">. {node.tip}</span>
              </button>
            ))}
            {/* En la fila de arriba la explicación se abre hacia abajo; hacia arriba la cortaría el borde del diagrama */}
            {hover && <div className={`tip${hover.y < 160 ? " below" : ""}`} aria-hidden="true" style={{ left: hover.x + (hover.w ?? 220) / 2, top: hover.y }}>{hover.tip}</div>}
          </div>
        </div>
      </div>
    </section>
    </div>
  );
}
