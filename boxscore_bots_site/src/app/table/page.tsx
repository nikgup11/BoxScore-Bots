'use client'; // Required for interactive features like search bars

import { useState } from 'react';
import Navbar from '@/components/nav_bar';

export default function StatsPage() {
  const [searchTerm, setSearchTerm] = useState('');

  const playerData = [
    { name: "LeBron James", points: 25.7, rebounds: 7.3, assists: 8.3 },
    { name: "Stephen Curry", points: 26.4, rebounds: 4.5, assists: 5.1 },
    { name: "Nikola Jokic", points: 26.1, rebounds: 12.4, assists: 9.0 },
    { name: "Luka Doncic", points: 33.9, rebounds: 9.2, assists: 9.8 },
  ];

  // Logic to filter the table based on the search bar
  const filteredPlayers = playerData.filter((player) =>
    player.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-[#87CEEB] font-sans text-[#2c3e50]">
      <Navbar />
      
      <main className="mx-auto max-w-5xl px-6 pt-24 pb-12">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white drop-shadow-md">
            PLAYER PROJECTIONS
          </h1>
          <p className="mt-2 text-lg font-medium">
            Projected performance metrics for the current season
          </p>
        </div>

        {/* --- Search Bar Section --- */}
        <div className="mb-8 flex justify-center">
          <input
            type="text"
            placeholder="Search for a player..."
            className="w-full max-w-md rounded-full border-2 border-white bg-white/90 px-6 py-3 shadow-lg outline-none transition-all focus:border-[#8B0000] focus:ring-2 focus:ring-[#8B0000]/20"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="overflow-hidden rounded-xl bg-white shadow-xl">
          <table className="w-full text-left border-collapse">
            <thead className="bg-[#8B0000] text-white">
              <tr>
                <th className="p-4 font-bold uppercase tracking-wider">Player Name</th>
                <th className="p-4 font-bold uppercase tracking-wider">Points</th>
                <th className="p-4 font-bold uppercase tracking-wider">Rebounds</th>
                <th className="p-4 font-bold uppercase tracking-wider">Assists</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-100">
              {filteredPlayers.length > 0 ? (
                filteredPlayers.map((player, index) => (
                  <tr key={index} className="hover:bg-blue-50 transition-colors">
                    <td className="p-4 font-semibold text-blue-900">{player.name}</td>
                    <td className="p-4">{player.points}</td>
                    <td className="p-4">{player.rebounds}</td>
                    <td className="p-4">{player.assists}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="p-8 text-center text-gray-500 italic">
                    No players found matching "{searchTerm}"
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
