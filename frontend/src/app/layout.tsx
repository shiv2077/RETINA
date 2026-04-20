import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import NavHeader from '@/components/NavHeader';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'RETINA — Visual Defect Detection',
  description: 'Adaptive two-stage defect detection system — Flanders Make / KU Leuven',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased min-h-screen flex flex-col">
        <NavHeader />

        <main className="flex-1 max-w-7xl mx-auto px-6 lg:px-8 py-8 w-full">
          {children}
        </main>

        <footer className="border-t border-surface-border mt-auto py-6">
          <div className="max-w-7xl mx-auto px-6 lg:px-8 flex justify-between items-center">
            <span className="text-text-tertiary text-xs">
              RETINA · Flanders Make · KU Leuven
            </span>
            <span className="text-text-tertiary text-xs">
              MPro + ProductionS Research Groups
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
