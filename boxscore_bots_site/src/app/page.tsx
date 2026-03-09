// app/page.tsx
import Image from 'next/image';
import Navbar from '@/components/nav_bar';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#87CEEB] font-sans text-[#2c3e50]">
      <Navbar />
      
      <div className="mx-auto flex min-h-[80vh] max-w-7xl flex-col items-center justify-between px-[10%] pt-[60px] md:flex-row">
        <main className="flex-1">
          <p className="mb-0 text-xl font-bold">say hello to...</p>
          <h1 className="mt-[5px] mb-[30px] text-5xl font-bold text-white drop-shadow-md">
            BOXSCORE BOT!
          </h1>
          <p className="max-w-[600px] text-xl leading-relaxed">
            BoxScore Bot is a Machine Learning powered
            sports line predictor. We train our ML models on
            public data to predict player performances over
            the season, and present their projected statistics
            in an easy-to-read format!
          </p>
        </main>

        <div className="mt-8 flex-1 md:mt-[8%] md:ml-[8%]">
          <Image 
            src="/bxb_mascot.png" 
            alt="Mascot"
            width={650}
            height={650}
            priority
            className="h-auto w-full max-w-[650px]"
          />
        </div>
      </div>
    </div>
  );
}
