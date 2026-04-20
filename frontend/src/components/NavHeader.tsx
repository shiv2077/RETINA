'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_LINKS = [
  { href: '/',                  label: 'Dashboard'   },
  { href: '/submit',            label: 'Submit'      },
  { href: '/label',             label: 'Review'      },
  { href: '/results',           label: 'Results'     },
  { href: '/demo',              label: 'Demo'        },
  { href: '/model-performance', label: 'Performance' },
];

export default function NavHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 h-14 flex items-center border-b border-surface-border bg-surface-base/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 w-full flex items-center justify-between">

        {/* Logo lockup */}
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 bg-kul-blue rounded-sm flex-shrink-0" />
          <span className="text-text-primary font-semibold text-sm tracking-wide">
            RETINA
          </span>
          <div className="w-px h-4 bg-surface-border mx-1" />
          <span className="text-text-tertiary text-xs hidden sm:block">
            Flanders Make
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-6">
          {NAV_LINKS.map(({ href, label }) => {
            const isActive = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={[
                  'text-sm transition-colors duration-150 pb-px',
                  isActive
                    ? 'text-text-primary border-b border-kul-accent'
                    : 'text-text-tertiary hover:text-text-primary',
                ].join(' ')}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* KU Leuven attribution */}
        <span className="text-text-tertiary text-xs hidden md:block">
          KU Leuven
        </span>
      </div>
    </header>
  );
}
