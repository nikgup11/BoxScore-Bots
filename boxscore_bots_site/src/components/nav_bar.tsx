// components/Navbar.tsx
import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 z-50 w-full bg-[#8B0000] shadow-md">
      <div className="mx-auto flex h-[60px] max-w-full items-center justify-between px-6">
        
        {/* Logo / Brand */}
        <Link
          href="/"
          className="text-lg font-extrabold tracking-wide text-white"
        >
          BOXSCORE BOTS
        </Link>

        {/* Navigation Links */}
        <ul className="flex items-center space-x-6 text-sm font-semibold text-white sm:space-x-10 sm:text-base">
          <li>
            <Link
              href="/table"
              className="transition-colors hover:text-[#ffcccc]"
            >
              Projection Table
            </Link>
          </li>

          <li>
            <Link
              href="/projection_tracking"
              className="transition-colors hover:text-[#ffcccc]"
            >
              Projection History
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}