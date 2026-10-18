import "@fontsource-variable/fraunces";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./styles.css";

import { lazy, StrictMode, Suspense, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";

import { Shell } from "./components/Layout";
import { ScrollTrigger } from "./scroll";

// Cada página se descarga al visitarla: Three.js solo en Inicio y Explorar, ECharts solo en Métricas
const Home = lazy(() => import("./pages/Home").then((m) => ({ default: m.Home })));
const AddPrompt = lazy(() => import("./pages/AddPrompt").then((m) => ({ default: m.AddPrompt })));
const Explore = lazy(() => import("./pages/Explore").then((m) => ({ default: m.Explore })));
const Metrics = lazy(() => import("./pages/Metrics").then((m) => ({ default: m.Metrics })));

const TITLES: Record<string, string> = {
  "/": "Premerie",
  "/agregar": "Agregar prompt · Premerie",
  "/explorar": "Cómo funciona el modelo · Premerie",
  "/metricas": "Métricas · Premerie",
};

function RouteEffects() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
    document.title = TITLES[pathname] ?? "Premerie";
    // Las secciones de la nueva página cambian las alturas: se recalculan los disparadores de scroll
    const id = requestAnimationFrame(() => ScrollTrigger.refresh());
    return () => cancelAnimationFrame(id);
  }, [pathname]);
  return null;
}

function NotFound() {
  return <div className="card"><h2>Esta página no existe</h2><Link to="/">Volver al inicio</Link></div>;
}

function App() {
  return (
    <BrowserRouter>
      <RouteEffects />
      <Suspense fallback={<div className="shell" aria-busy="true" />}>
      <Routes>
        <Route path="/" element={<Shell><Home /></Shell>} />
        <Route path="/agregar" element={<Shell><AddPrompt /></Shell>} />
        <Route path="/metricas" element={<Shell><Metrics /></Shell>} />
        <Route path="/explorar" element={<Shell flow><Explore /></Shell>} />
        <Route path="*" element={<Shell><NotFound /></Shell>} />
      </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
