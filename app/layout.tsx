import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Kikitori — spike",
  description: "G0 technical spike: kotoba-whisper ONNX on WebGPU",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body style={{ fontFamily: "ui-monospace, monospace", margin: 16 }}>
        {children}
      </body>
    </html>
  );
}
