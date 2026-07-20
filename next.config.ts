import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Kikifude is a fully client-side tool (charter §3): static export, no backend.
  output: "export",

  // Dev-server only: has no effect on `next build` / static export output.
  // Next.js blocks cross-origin requests to /_next/* dev resources (HMR etc.) by
  // default, which breaks hydration when a phone on the LAN opens the dev server
  // by IP (`next dev -H 0.0.0.0`). WebGPU needs a secure context, so real-device
  // testing must go through the HTTPS dev server rather than a plain LAN URL.
  // The wildcard covers DHCP reassignment within the 192.168.0.0/16 private range;
  // each "*" matches exactly one dot-separated segment, so public hosts are still
  // rejected. Use a different range here if your LAN is not 192.168.x.x.
  allowedDevOrigins: ["192.168.*.*"],
};

export default nextConfig;
