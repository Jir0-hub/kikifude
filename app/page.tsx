import Link from "next/link";

export default function Home() {
  return (
    <main>
      <h1>Kikitori (spike repo)</h1>
      <p>
        <Link href="/spike">→ /spike (G0 benchmark)</Link>
      </p>
    </main>
  );
}
