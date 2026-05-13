import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EchoGhost Hub Ultra",
  description: "Multi-Mode RF Sensing Platform with HackRF One",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
