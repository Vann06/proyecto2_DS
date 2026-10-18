// Genera los modelos 3D de la interfaz (public/models/*.glb) comprimidos con Draco.
// Uso: node scripts/build-models.mjs
// La geometría se arma con Three.js y se escribe con glTF-Transform; no hace falta un programa de modelado.
import { Document, NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { draco, weld } from "@gltf-transform/functions";
import draco3d from "draco3dgltf";
import { mkdirSync } from "node:fs";
import * as THREE from "three";

const OUT = new URL("../public/models/", import.meta.url);
mkdirSync(OUT, { recursive: true });

function material(doc, name, color, roughness = 0.8, metallic = 0) {
  const c = new THREE.Color(color).convertSRGBToLinear();
  return doc.createMaterial(name).setBaseColorFactor([c.r, c.g, c.b, 1]).setRoughnessFactor(roughness)
    .setMetallicFactor(metallic).setDoubleSided(true);
}

// Convierte una BufferGeometry en una malla de glTF
function mesh(doc, buffer, name, geometry, mat) {
  const g = geometry.index ? geometry.toNonIndexed() : geometry;
  g.computeVertexNormals();
  const accessor = (array, type) => doc.createAccessor().setArray(new Float32Array(array)).setType(type).setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setAttribute("POSITION", accessor(g.attributes.position.array, "VEC3"))
    .setAttribute("NORMAL", accessor(g.attributes.normal.array, "VEC3"))
    .setMaterial(mat);
  return doc.createMesh(name).addPrimitive(primitive);
}

function node(doc, scene, buffer, name, geometry, mat, position = [0, 0, 0], rotation = [0, 0, 0]) {
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation));
  const n = doc.createNode(name).setMesh(mesh(doc, buffer, name, geometry, mat)).setTranslation(position)
    .setRotation([q.x, q.y, q.z, q.w]);
  scene.addChild(n);
  return n;
}

function gearGeometry(radius, teeth, depth) {
  const shape = new THREE.Shape();
  const steps = teeth * 4;
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    const r = i % 4 < 2 ? radius : radius * 0.82;
    const x = Math.cos(angle) * r;
    const y = Math.sin(angle) * r;
    i === 0 ? shape.moveTo(x, y) : shape.lineTo(x, y);
  }
  const hole = new THREE.Path();
  hole.absarc(0, 0, radius * 0.22, 0, Math.PI * 2, true);
  shape.holes.push(hole);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments: 8 });
  geometry.translate(0, 0, -depth / 2);
  return geometry;
}

async function write(doc, file) {
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    "draco3d.encoder": await draco3d.createEncoderModule(),
    "draco3d.decoder": await draco3d.createDecoderModule(),
  });
  await doc.transform(weld(), draco({ method: "edgebreaker" }));
  await io.write(new URL(file, OUT).pathname.replace(/^\/([A-Za-z]:)/, "$1"), doc);
  console.log("escrito", file);
}

// Hoja de papel ligeramente curvada, con renglones en relieve
async function paper() {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const scene = doc.createScene("hoja");
  const sheet = new THREE.PlaneGeometry(1, 1.414, 16, 20);
  const pos = sheet.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    pos.setZ(i, 0.045 * Math.sin((x + 0.5) * Math.PI) + 0.02 * Math.sin(y * 3));
  }
  node(doc, scene, buffer, "sheet", sheet, material(doc, "papel", "#F6E7CC", 0.95));
  const ink = material(doc, "tinta", "#B9875A", 0.9);
  for (let row = 0; row < 9; row++) {
    const width = row === 0 ? 0.45 : 0.62 + ((row * 37) % 10) / 60;
    const line = new THREE.BoxGeometry(width, 0.018, 0.004);
    const x = -0.5 + 0.12 + width / 2;
    const y = 0.52 - row * 0.11;
    node(doc, scene, buffer, `line${row}`, line, ink, [x, y, 0.045 * Math.sin((x + 0.5) * Math.PI) + 0.02 * Math.sin(y * 3) + 0.004]);
  }
  await write(doc, "paper.glb");
}

// La "máquina" de LightGBM: tolva, cuerpo, dos engranes, lámpara y dos bandejas de salida
async function machine() {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const scene = doc.createScene("maquina");
  const wood = material(doc, "madera", "#8A5426", 0.75);
  const dark = material(doc, "tinta", "#2E1F12", 0.6);
  const brass = material(doc, "laton", "#C9974F", 0.35, 0.8);
  const cream = material(doc, "papel", "#F4E4CC", 0.9);

  node(doc, scene, buffer, "body", new THREE.BoxGeometry(2.2, 1.3, 1.2), wood, [0, 0, 0]);
  node(doc, scene, buffer, "panel", new THREE.BoxGeometry(1.7, 0.9, 0.04), cream, [0, 0.02, 0.62]);
  node(doc, scene, buffer, "hopper", new THREE.CylinderGeometry(0.62, 0.22, 0.7, 24, 1, true), brass, [0, 1.0, 0]);
  node(doc, scene, buffer, "gearA", gearGeometry(0.34, 12, 0.08), brass, [-0.38, 0.08, 0.68]);
  node(doc, scene, buffer, "gearB", gearGeometry(0.24, 9, 0.08), dark, [0.12, 0.2, 0.7]);
  node(doc, scene, buffer, "gearC", gearGeometry(0.2, 8, 0.08), brass, [0.48, -0.12, 0.68]);
  node(doc, scene, buffer, "lamp", new THREE.SphereGeometry(0.09, 16, 12), brass, [0.82, 0.48, 0.64]);
  node(doc, scene, buffer, "trayContent", new THREE.BoxGeometry(0.6, 0.06, 0.7), dark, [-0.7, -0.62, 0.9], [0.25, 0, 0]);
  node(doc, scene, buffer, "trayWording", new THREE.BoxGeometry(0.6, 0.06, 0.7), dark, [0.7, -0.62, 0.9], [0.25, 0, 0]);
  for (const x of [-0.95, 0.95]) {
    node(doc, scene, buffer, `foot${x > 0 ? "R" : "L"}`, new THREE.CylinderGeometry(0.1, 0.12, 0.2, 12), dark, [x, -0.75, 0.4]);
  }
  await write(doc, "machine.glb");
}

await paper();
await machine();
