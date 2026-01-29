import type { Metadata } from 'next';
import './globals.css';

/**
 * RETINA Frontend - Root Layout
 * ==============================
 *
 * This is the root layout for the entire application.
 * It provides:
 * - Global styles
 * - Navigation header
 * - Footer with project info
 */

export const metadata: Metadata = {
  title: 'RETINA - Multi-Stage Visual Anomaly Detection',
  description:
    'Research system for industrial visual inspection using active learning and multi-stage anomaly detection',
  keywords: ['anomaly detection', 'machine learning', 'active learning', 'industrial inspection'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Navigation Header */}
        <header className="bg-blue-900 text-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo and Title */}
              <div className="flex items-center space-x-3">
                <div className="text-2xl font-bold">👁️ RETINA</div>
                <span className="hidden sm:block text-sm text-blue-200">
                  Multi-Stage Visual Anomaly Detection
                </span>
              </div>

              {/* Navigation Links */}
              <nav className="flex items-center space-x-4">
                <NavLink href="/">Dashboard</NavLink>
                <NavLink href="/submit">Submit</NavLink>
                <NavLink href="/results">Results</NavLink>
                <NavLink href="/label">Label</NavLink>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="bg-gray-100 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row justify-between items-center text-sm text-gray-600 dark:text-gray-400">
              <div>
                <span className="font-medium">RETINA</span>
                <span className="mx-2">•</span>
                <span>Anomaly Detection System</span>
              </div>
              <div className="mt-2 sm:mt-0">
                <span className="text-xs">
                  Multi-Stage Visual Anomaly Detection with Active Learning
                </span>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}

/**
 * Navigation link component with hover styling.
 */
function NavLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      className="px-3 py-2 rounded-md text-sm font-medium text-blue-100 hover:text-white hover:bg-blue-800 transition-colors"
    >
      {children}
    </a>
  );
}
