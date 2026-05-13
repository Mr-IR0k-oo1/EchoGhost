"use client";

/**
 * EchoGhost3D — Three.js 3D visualization component.
 *
 * Renders a 3D room scene with:
 *   - Transparent room wireframe box
 *   - Radar position indicator (HackRF)
 *   - Colored motion particles driven by radar heatmap data
 *   - Range rings and grid floor
 *   - Optional skeleton / target blips
 *
 * Motion particles are spawned/updated based on real-time heatmap
 * and range-Doppler data from the WebSocket stream.
 */

import { useRef, useMemo, useEffect, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";

// ── Types ─────────────────────────────────────────────────────────────────

interface ParticleData {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  color: THREE.Color;
  size: number;
  life: number;
  maxLife: number;
}

interface EchoGhost3DProps {
  heatmap?: number[][];
  motionMagnitude?: number;
  targetRange?: number;
  breathingRate?: number;
  activityLabel?: string;
  mode?: string;
}

// ── Constants ─────────────────────────────────────────────────────────────

const ROOM_SIZE = 10; // meters (visualized)
const MAX_PARTICLES = 500;
const COLORS = {
  active: new THREE.Color("#00f5ff"),
  motion: new THREE.Color("#ff00e5"),
  breathing: new THREE.Color("#00ff88"),
  idle: new THREE.Color("#555577"),
  danger: new THREE.Color("#ff3355"),
};

// ── Particle System ───────────────────────────────────────────────────────

function ParticleSystem({ heatmap, motionMagnitude = 0 }: {
  heatmap?: number[][];
  motionMagnitude?: number;
}) {
  const particlesRef = useRef<ParticleData[]>([]);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const colorDummy = useMemo(() => new THREE.Color(), []);

  // Update particles based on incoming heatmap data
  useEffect(() => {
    if (!heatmap || heatmap.length === 0) return;

    const rows = heatmap.length;
    const cols = heatmap[0]?.length ?? rows;

    // Spawn particles at high-energy heatmap positions
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const energy = heatmap[r][c];
        if (energy > 0.3 && particlesRef.current.length < MAX_PARTICLES) {
          // Map heatmap coordinates to 3D room space
          const x = (c / cols - 0.5) * ROOM_SIZE;
          const z = (r / rows - 0.5) * ROOM_SIZE;
          const y = Math.random() * 2;

          const color = new THREE.Color().setHSL(
            0.5 - energy * 0.4,
            0.8,
            0.3 + energy * 0.5
          );

          particlesRef.current.push({
            position: new THREE.Vector3(x, y, z),
            velocity: new THREE.Vector3(
              (Math.random() - 0.5) * 0.5,
              Math.random() * 0.3,
              (Math.random() - 0.5) * 0.5
            ),
            color,
            size: 0.05 + energy * 0.15,
            life: 1.0,
            maxLife: 1.0 + Math.random() * 2,
          });
        }
      }
    }

    // Add motion burst particles
    if (motionMagnitude > 0.3) {
      for (let i = 0; i < Math.floor(motionMagnitude * 20); i++) {
        if (particlesRef.current.length >= MAX_PARTICLES) break;
        const angle = Math.random() * Math.PI * 2;
        const speed = 0.5 + motionMagnitude * 2;
        particlesRef.current.push({
          position: new THREE.Vector3(0, 0.5, 0),
          velocity: new THREE.Vector3(
            Math.cos(angle) * speed,
            Math.random() * speed,
            Math.sin(angle) * speed
          ),
          color: COLORS.motion.clone(),
          size: 0.08,
          life: 1.0,
          maxLife: 0.5 + Math.random() * 0.5,
        });
      }
    }
  }, [heatmap, motionMagnitude]);

  // Animate particles each frame
  useFrame((_, delta) => {
    if (!meshRef.current) return;

    const particles = particlesRef.current;
    const count = particles.length;

    // Update and cull
    let writeIdx = 0;
    for (let i = 0; i < count; i++) {
      const p = particles[i];
      p.life -= delta / p.maxLife;

      if (p.life <= 0) continue; // Remove dead particles

      // Apply physics
      p.position.x += p.velocity.x * delta;
      p.position.y += p.velocity.y * delta;
      p.position.z += p.velocity.z * delta;
      p.velocity.y -= 0.5 * delta; // Gravity

      // Fade color
      p.color.multiplyScalar(0.995);

      // Boundary wrap / bounce
      const halfRoom = ROOM_SIZE / 2;
      if (Math.abs(p.position.x) > halfRoom) p.velocity.x *= -1;
      if (Math.abs(p.position.z) > halfRoom) p.velocity.z *= -1;
      if (p.position.y < 0) {
        p.position.y = 0;
        p.velocity.y *= -0.3;
      }

      particles[writeIdx] = p;
      writeIdx++;
    }

    particles.length = writeIdx;

    // Update instanced mesh
    for (let i = 0; i < writeIdx; i++) {
      const p = particles[i];
      dummy.position.copy(p.position);
      dummy.scale.setScalar(p.size);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      meshRef.current.setColorAt(i, p.color);
    }

    meshRef.current.count = writeIdx;
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_PARTICLES]}
      count={0}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial transparent opacity={0.8} depthWrite={false} />
    </instancedMesh>
  );
}

// ── Room Grid ─────────────────────────────────────────────────────────────

function RoomGrid() {
  const gridSize = ROOM_SIZE;
  const divisions = 10;

  const gridLines = useMemo(() => {
    const lines: { key: string; start: THREE.Vector3; end: THREE.Vector3 }[] = [];
    const half = gridSize / 2;
    const step = gridSize / divisions;

    for (let i = 0; i <= divisions; i++) {
      const pos = -half + i * step;
      lines.push({
        key: `h${i}`,
        start: new THREE.Vector3(-half, 0, pos),
        end: new THREE.Vector3(half, 0, pos),
      });
      lines.push({
        key: `v${i}`,
        start: new THREE.Vector3(pos, 0, -half),
        end: new THREE.Vector3(pos, 0, half),
      });
    }
    return lines;
  }, []);

  return (
    <group>
      {gridLines.map((line) => (
        <Line
          key={line.key}
          points={[line.start, line.end]}
          color="#2a2a3e"
          lineWidth={1}
          transparent
          opacity={0.5}
        />
      ))}
    </group>
  );
}

// ── Range Rings ───────────────────────────────────────────────────────────

function RangeRings({ targetRange = 0 }: { targetRange?: number }) {
  const rings = [2, 5, 8];

  return (
    <group>
      {rings.map((radius) => (
        <mesh
          key={radius}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0.01, 0]}
        >
          <ringGeometry args={[radius - 0.05, radius + 0.05, 64]} />
          <meshBasicMaterial
            color={targetRange > radius - 1 && targetRange < radius + 1
              ? "#00f5ff"
              : "#1a1a2e"}
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
      {targetRange > 0 && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <ringGeometry
            args={[targetRange - 0.1, targetRange + 0.1, 64]}
          />
          <meshBasicMaterial
            color="#ff00e5"
            transparent
            opacity={0.6}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </group>
  );
}

// ── Room Wireframe ────────────────────────────────────────────────────────

function RoomWireframe() {
  const half = ROOM_SIZE / 2;
  const corners = [
    [-half, 0, -half],
    [half, 0, -half],
    [half, 0, half],
    [-half, 0, half],
    [-half, 0, -half],
    [-half, 2.5, -half],
    [half, 2.5, -half],
    [half, 2.5, half],
    [-half, 2.5, half],
    [-half, 2.5, -half],
  ];

  const verticals = [
    [half, 0, -half, half, 2.5, -half],
    [half, 0, half, half, 2.5, half],
    [-half, 0, half, -half, 2.5, half],
  ];

  return (
    <group>
      {corners.length > 1 && (
        <Line
          points={corners.map((c) => new THREE.Vector3(c[0], c[1], c[2]))}
          color="#2a2a3e"
          lineWidth={1}
          transparent
          opacity={0.6}
        />
      )}
      {verticals.map((v, i) => (
        <Line
          key={i}
          points={[
            new THREE.Vector3(v[0], v[1], v[2]),
            new THREE.Vector3(v[3], v[4], v[5]),
          ]}
          color="#2a2a3e"
          lineWidth={1}
          transparent
          opacity={0.6}
        />
      ))}
    </group>
  );
}

// ── Radar Device Indicator ────────────────────────────────────────────────

function RadarDevice() {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = clock.getElapsedTime() * 0.5;
    }
  });

  return (
    <group position={[0, 0.15, 0]}>
      <mesh ref={ref}>
        <boxGeometry args={[0.15, 0.08, 0.15]} />
        <meshStandardMaterial
          color="#00f5ff"
          emissive="#00f5ff"
          emissiveIntensity={0.5}
        />
      </mesh>
      {/* Antenna indicator */}
      <mesh position={[0, 0.12, 0]}>
        <cylinderGeometry args={[0.01, 0.02, 0.1, 8]} />
        <meshBasicMaterial color="#00f5ff" transparent opacity={0.6} />
      </mesh>
      {/* TX/RX sweep visualization */}
      <mesh>
        <ringGeometry args={[0.2, 0.25, 32]} />
        <meshBasicMaterial
          color="#00f5ff"
          transparent
          opacity={0.15}
          side={THREE.DoubleSide}
        />
      </mesh>
      <Text
        position={[0, -0.15, 0]}
        fontSize={0.12}
        color="#555577"
        anchorX="center"
        anchorY="top"
      >
        HACKRF ONE
      </Text>
    </group>
  );
}

// ── Breathing Visualization ───────────────────────────────────────────────

function BreathingRing({ rate = 0 }: { rate?: number }) {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (ref.current && rate > 0) {
      const phase = Math.sin(clock.getElapsedTime() * rate * Math.PI / 30);
      ref.current.scale.setScalar(1 + phase * 0.1);
      ref.current.material.opacity = 0.2 + Math.abs(phase) * 0.3;
    }
  });

  if (rate <= 0) return null;

  return (
    <mesh
      ref={ref}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0.05, 2]}
    >
      <ringGeometry args={[0.3, 0.4, 32]} />
      <meshBasicMaterial
        color="#00ff88"
        transparent
        opacity={0.3}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ── Activity Label ────────────────────────────────────────────────────────

function ActivityLabel({ label = "idle" }: { label?: string }) {
  const labelColors: Record<string, string> = {
    idle: "#555577",
    walking: "#00f5ff",
    sitting: "#00ff88",
    gesturing: "#ff00e5",
    falling: "#ff3355",
  };

  const color = labelColors[label] ?? "#555577";

  return (
    <Text
      position={[0, 2.8, -ROOM_SIZE / 2 + 0.5]}
      fontSize={0.2}
      color={color}
      anchorX="center"
      anchorY="top"
    >
      {label.toUpperCase()}
    </Text>
  );
}

// ── Scene ─────────────────────────────────────────────────────────────────

function Scene({
  heatmap,
  motionMagnitude,
  targetRange,
  breathingRate,
  activityLabel,
}: EchoGhost3DProps) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.2} />
      <pointLight position={[5, 5, 5]} intensity={0.5} color="#00f5ff" />
      <pointLight position={[-5, 3, -5]} intensity={0.3} color="#ff00e5" />

      {/* Grid floor */}
      <RoomGrid />
      <RangeRings targetRange={targetRange} />

      {/* Room */}
      <RoomWireframe />

      {/* Radar device */}
      <RadarDevice />

      {/* Particles */}
      <ParticleSystem heatmap={heatmap} motionMagnitude={motionMagnitude} />

      {/* Breathing */}
      <BreathingRing rate={breathingRate} />

      {/* Activity label */}
      <ActivityLabel label={activityLabel} />

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={2}
        maxDistance={20}
        maxPolarAngle={Math.PI / 2.1}
      />
    </>
  );
}

// ── Camera Setup ──────────────────────────────────────────────────────────

function CameraController() {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(6, 4, 6);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return null;
}

// ── Main Export ───────────────────────────────────────────────────────────

export default function EchoGhost3D(props: EchoGhost3DProps) {
  return (
    <div className="w-full h-full rounded-xl overflow-hidden">
      <Canvas
        className="echoghost-canvas"
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: "high-performance",
        }}
      >
        <CameraController />
        <Scene {...props} />
      </Canvas>
    </div>
  );
}
