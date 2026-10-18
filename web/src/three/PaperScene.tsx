// Hojas de papel que flotan y, al bajar, se acomodan en una pila (pantalla de inicio).
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import type { Group, Object3D } from "three";

import type { Smoothed } from "../scroll";
import { lerp } from "../scroll";
import { useCompressedGLTF } from "./gltf";

interface Pointer {
  x: Smoothed;
  y: Smoothed;
}

// Posición flotante y posición en la pila de cada hoja
const SHEETS = [
  { float: [-2.3, 0.4, -0.6], rot: [0.3, 0.5, -0.4], stack: 0 },
  { float: [-1.0, -0.5, 0.4], rot: [-0.2, -0.3, 0.25], stack: 1 },
  { float: [0.4, 0.6, -0.2], rot: [0.15, 0.4, 0.1], stack: 2 },
  { float: [1.6, -0.3, 0.3], rot: [-0.3, -0.5, -0.3], stack: 3 },
  { float: [2.6, 0.5, -0.8], rot: [0.25, 0.2, 0.45], stack: 4 },
] as const;

export function PaperScene({ progress, pointer }: { progress: { current: Smoothed }; pointer: { current: Pointer } }) {
  const { scene } = useCompressedGLTF("/models/paper.glb");
  const sheets = useMemo(() => SHEETS.map(() => scene.clone(true)), [scene]);
  const group = useRef<Group>(null);
  const refs = useRef<(Object3D | null)[]>([]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const p = progress.current.value;
    SHEETS.forEach((sheet, i) => {
      const object = refs.current[i];
      if (!object) return;
      const bob = Math.sin(t * 0.8 + i * 1.3) * 0.12 * (1 - p);
      object.position.set(
        lerp(sheet.float[0], 0.08 * (i - 2), p),
        lerp(sheet.float[1] + bob, -0.55 + sheet.stack * 0.05, p),
        lerp(sheet.float[2], 0, p),
      );
      object.rotation.set(
        lerp(sheet.rot[0] + Math.sin(t * 0.5 + i) * 0.08, -Math.PI / 2 + 0.02, p),
        lerp(sheet.rot[1], 0, p),
        lerp(sheet.rot[2], (i - 2) * 0.06, p),
      );
    });
    if (group.current) {
      group.current.rotation.y = pointer.current.x.value * 0.18;
      group.current.rotation.x = pointer.current.y.value * 0.08;
    }
  });

  return (
    <group ref={group} scale={0.95}>
      {sheets.map((object, i) => (
        <primitive key={i} object={object} ref={(node: Object3D | null) => { refs.current[i] = node; }} />
      ))}
    </group>
  );
}
