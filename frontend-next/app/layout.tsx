import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReSearch — AI Research Assistant",
  description: "Ask research questions, get answers grounded in academic papers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
