// components/Navbar.tsx
import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 z-50 flex h-[50px] w-full items-center justify-end bg-[#8B0000] px-[5%]">
      <ul className="mr-[100px] flex gap-[500px] list-none">
        <li>
          <Link href="/page1" className="text-base font-bold text-white transition-colors hover:text-[#ffcccc]">
            Page 1
          </Link>
        </li>
        <li>
          <Link href="/page3" className="text-base font-bold text-white transition-colors hover:text-[#ffcccc]">
            Page 2
          </Link>
        </li>
      </ul>
    </nav>
  );
}
