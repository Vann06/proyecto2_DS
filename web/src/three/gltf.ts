// Carga de modelos glTF comprimidos: geometría con Draco y texturas con KTX2 (Basis), con los decodificadores
// servidos desde public/ para no depender de un CDN.
import { useGLTF } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import type { WebGLRenderer } from "three";
// GLTFLoader de drei viene de three-stdlib, así que el KTX2Loader también
import { KTX2Loader } from "three-stdlib";

export const DRACO_PATH = "/draco/";
export const MAX_DPR = 2;

let ktx2: KTX2Loader | null = null;

function ktx2Loader(gl: WebGLRenderer) {
  if (!ktx2) ktx2 = new KTX2Loader().setTranscoderPath("/basis/").detectSupport(gl);
  return ktx2;
}

export function useCompressedGLTF(url: string) {
  const gl = useThree((state) => state.gl);
  return useGLTF(url, DRACO_PATH, false, (loader) => {
    loader.setKTX2Loader(ktx2Loader(gl));
  });
}
