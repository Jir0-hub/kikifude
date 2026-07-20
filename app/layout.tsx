import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Otofude（オトフデ）— ブラウザ完結・無料の日本語文字起こし",
  description:
    "音声をサーバーに送信せず、すべてブラウザ内（WebGPU）で処理する無料の日本語文字起こしツール。",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
