import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Kikitori is a fully client-side tool (charter §3): static export, no backend.
  output: "export",
};

export default nextConfig;
