// Marco de todas las pantallas: barra lateral fija que se puede ocultar. Las pantallas estáticas llevan además
// el degradado con su oleaje; Cómo funciona (página de scroll) usa el fondo de papel liso.
import { type ReactNode, useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

import { ScrollTrigger } from "../scroll";

const LINKS = [
  { to: "/", label: "Inicio", icon: <><path d="M3 9.5L10 4l7 5.5" /><path d="M5 8.5V16h10V8.5" /></> },
  { to: "/agregar", label: "Agregar prompt", icon: <path d="M10 4v12M4 10h12" /> },
  { to: "/explorar", label: "Cómo funciona", icon: <><circle cx="9" cy="9" r="5.5" /><path d="M13 13l4 4" /></> },
  { to: "/metricas", label: "Métricas", icon: <path d="M4 16V9M9 16V4M14 16v-5" /> },
];

const STORAGE_KEY = "premerie-sidebar";

export function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <path d="M7 5h14l6 6v18H7z" />
      <path d="M21 5v6h6" />
      <path d="M11 17h12M11 21h12M11 25h7" strokeLinecap="round" />
    </svg>
  );
}

/** Tres manchas desenfocadas que se mueven despacio sobre el degradado, como un oleaje. */
export function Waves() {
  return (
    <div className="waves" aria-hidden="true">
      <div className="wave wave-1" />
      <div className="wave wave-2" />
      <div className="wave wave-3" />
    </div>
  );
}

function readCollapsed() {
  try {
    return localStorage.getItem(STORAGE_KEY) === "collapsed";
  } catch {
    return false;
  }
}

export function Shell({ children, flow = false }: { children: ReactNode; flow?: boolean }) {
  const [collapsed, setCollapsed] = useState(readCollapsed);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? "collapsed" : "open");
    } catch {
      // Sin almacenamiento (ventana privada): el estado solo dura esta visita
    }
    // El contenido cambia de ancho al terminar la transición: se recalculan los disparadores de scroll
    const id = window.setTimeout(() => ScrollTrigger.refresh(), 320);
    return () => window.clearTimeout(id);
  }, [collapsed]);

  return (
    <div className={`shell${flow ? " flow" : ""}`}>
      <a href="#contenido" className="skip-link">Saltar al contenido</a>
      {!flow && <Waves />}
      <nav className={`sidebar${collapsed ? " collapsed" : ""}`} aria-label="Navegación principal">
        <div className="sidebar-head">
          <NavLink to="/" className="brand" aria-label="Premerie, inicio"><Logo /><span className="brand-word">Premerie</span></NavLink>
          <button type="button" className="icon-btn sidebar-toggle" aria-expanded={!collapsed} aria-controls="nav-links"
            aria-label={collapsed ? "Mostrar menú" : "Ocultar menú"} onClick={() => setCollapsed(!collapsed)}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
              <rect x="3" y="4" width="14" height="12" rx="2" /><path d="M8 4v12" />
            </svg>
          </button>
        </div>
        <div className="nav" id="nav-links">
          {LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} end title={collapsed ? link.label : undefined}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8"
                strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{link.icon}</svg>
              <span className="nav-label">{link.label}</span>
            </NavLink>
          ))}
        </div>
      </nav>
      <main id="contenido" tabIndex={-1} className={flow ? "content-flow" : "content"}>{children}</main>
    </div>
  );
}
