// Aparece suavemente al entrar en pantalla, siguiendo el scroll (no una animación de duración fija).
import { type ReactNode, useRef } from "react";

import { useScrollProgress } from "../scroll";

export function Reveal({ children, className = "", delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  const box = useRef<HTMLDivElement>(null);
  useScrollProgress(box, {
    // clamp(): al final de la página el scroll no alcanza a subir el último elemento hasta 62 %, y se quedaría a
    // medio aparecer
    start: "clamp(top 92%)",
    end: "clamp(top 62%)",
    onChange: (p) => {
      if (!box.current) return;
      const shown = Math.min(1, Math.max(0, (p - delay) / (1 - delay)));
      box.current.style.opacity = `${shown}`;
      box.current.style.transform = `translateY(${(1 - shown) * 24}px)`;
    },
  });
  return <div ref={box} className={`reveal ${className}`}>{children}</div>;
}
