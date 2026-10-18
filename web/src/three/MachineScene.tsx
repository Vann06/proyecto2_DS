// La "máquina" de LightGBM: los engranes giran con el scroll y la lámpara se enciende al salir las notas.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { Color, type Group, type Mesh, MeshStandardMaterial } from "three";

import type { Smoothed } from "../scroll";
import { useCompressedGLTF } from "./gltf";

const LAMP_ON = new Color("#FFB45A");
const BLACK = new Color("#000000");

export function MachineScene({ progress }: { progress: { current: Smoothed } }) {
  const { scene, nodes } = useCompressedGLTF("/models/machine.glb");
  const group = useRef<Group>(null);
  const lampMaterial = useMemo(() => new MeshStandardMaterial({ color: "#C9974F", roughness: 0.3, metalness: 0.6 }), []);

  useEffect(() => {
    const lamp = nodes.lamp as Mesh | undefined;
    if (lamp) lamp.material = lampMaterial;
    return () => lampMaterial.dispose();
  }, [nodes, lampMaterial]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const p = progress.current.value;
    const turn = p * Math.PI * 4 + t * 0.15;
    if (nodes.gearA) nodes.gearA.rotation.z = turn;
    if (nodes.gearB) nodes.gearB.rotation.z = -turn * (12 / 9) + 0.2;
    if (nodes.gearC) nodes.gearC.rotation.z = turn * (12 / 8);
    const glow = Math.max(0, (p - 0.75) / 0.25);
    lampMaterial.emissive.copy(BLACK).lerp(LAMP_ON, glow);
    lampMaterial.emissiveIntensity = 2 * glow;
    if (nodes.trayContent) nodes.trayContent.position.z = 0.9 + 0.15 * glow;
    if (nodes.trayWording) nodes.trayWording.position.z = 0.9 + 0.15 * glow;
    if (group.current) {
      group.current.rotation.y = -0.5 + p * 0.7 + Math.sin(t * 0.4) * 0.03;
      group.current.position.y = Math.sin(t * 1.2) * 0.04;
    }
  });

  return (
    <group ref={group} rotation={[0.18, -0.5, 0]}>
      <primitive object={scene} />
    </group>
  );
}
