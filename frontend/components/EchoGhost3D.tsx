"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, Text } from "@react-three/drei";
import * as THREE from "three";
import { useWs } from "./WebSocketProvider";

function Room() {
  return (
    <mesh>
      <boxGeometry args={[10, 6, 10]} />
      <meshStandardMaterial
        wireframe
        color="#334455"
        transparent
        opacity={0.3}
      />
    </mesh>
  );
}

function Floor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3, 0]}>
      <planeGeometry args={[12, 12]} />
      <meshStandardMaterial
        color="#0a1628"
        metalness={0.6}
        roughness={0.4}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
}

function MotionParticles() {
  const { frame } = useWs();
  const groupRef = useRef<THREE.Group>(null);

  const positions = useMemo(() => {
    if (!frame?.positions?.length) {
      return [{ x: 0, y: 0, z: 0, intensity: 0, label: "idle" }];
    }
    return frame.positions;
  }, [frame?.positions]);

  const motionScore = frame?.motion?.score ?? 0;

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <group ref={groupRef}>
      {positions.map((pos, i) => {
        const hue = pos.intensity * 240;
        return (
          <mesh key={i} position={[pos.x, pos.y, pos.z]}>
            <sphereGeometry
              args={[0.08 + pos.intensity * 0.2, 16, 16]}
            />
            <meshStandardMaterial
              color={`hsl(${hue}, 100%, 60%)`}
              emissive={`hsl(${hue}, 100%, 40%)`}
              emissiveIntensity={0.5 + motionScore * 5}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function GridHelper() {
  return <gridHelper args={[12, 12, "#1a2a4a", "#0d1a2d"]} />;
}

function AxisLabels() {
  return (
    <>
      <Text position={[5.5, -2.5, 0]} fontSize={0.3} color="#445566">
        Range
      </Text>
      <Text position={[0, -2.5, 5.5]} fontSize={0.3} color="#445566">
        Cross
      </Text>
    </>
  );
}

export default function EchoGhost3D() {
  return (
    <div className="w-full h-full bg-[#050a18]">
      <Canvas
        camera={{ position: [0, 8, 12], fov: 50 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1.0} />
        <pointLight position={[-10, -5, -10]} intensity={0.3} color="#4466ff" />

        <Room />
        <Floor />
        <GridHelper />
        <AxisLabels />
        <MotionParticles />

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          minDistance={4}
          maxDistance={25}
        />
        <Environment preset="night" />
      </Canvas>
    </div>
  );
}
