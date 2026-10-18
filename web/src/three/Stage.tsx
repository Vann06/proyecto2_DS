// Lienzo de R3F común: resolución limitada a 2x y sin dibujar mientras está fuera de la pantalla.
import { Canvas } from "@react-three/fiber";
import { type ReactNode, Suspense, useEffect, useRef, useState } from "react";

import { MAX_DPR } from "./gltf";

interface Props {
  children: ReactNode;
  camera?: { position: [number, number, number]; fov?: number };
  className?: string;
  label: string;
}

export function Stage({ children, camera = { position: [0, 0, 5], fov: 40 }, className, label }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!box.current) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: "120px" });
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={box} className={className} role="img" aria-label={label}>
      {/* offsetSize: mide la caja sin la escala CSS del diagrama; con el tamaño ya escalado el lienzo queda más
          grande que su caja y la máquina se ve cortada en pantallas grandes */}
      <Canvas dpr={[1, MAX_DPR]} camera={camera} frameloop={visible ? "always" : "never"} resize={{ offsetSize: true }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}>
        <ambientLight intensity={1.1} />
        <directionalLight position={[3, 5, 4]} intensity={2.2} color="#FFF1DC" />
        <directionalLight position={[-4, -2, 2]} intensity={0.6} color="#D2985F" />
        <Suspense fallback={null}>{children}</Suspense>
      </Canvas>
    </div>
  );
}
