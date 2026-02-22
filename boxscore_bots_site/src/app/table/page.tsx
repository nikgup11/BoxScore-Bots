'use client';

import { useState, useEffect } from 'react';
import Navbar from '@/components/nav_bar';
import { supabase } from '../../lib/supabase'; // Relative path to lib

type Player = {
  name: string;
  team: string;
  opponent: string;
  points: number | null;
  rebounds: number | null;
  assists: number | null;
  date: string | null;
};

export default function StatsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [playerData, setPlayerData] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch data from Supabase
  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data, error } = await supabase
          .from('projections')
          .select('name, team, opp, proj_pts, proj_reb, proj_ast, game_date');

        if (error) {
          console.error('Error fetching projections:', error.message);
          setPlayerData([]);
        } else {
          // Map DB columns to frontend type
          const mapped = (data || []).map((row: any) => ({
            name: row.name,
            team: row.team,
            opponent: row.opp,
            points: row.proj_pts,
            rebounds: row.proj_reb,
            assists: row.proj_ast,
            date: row.game_date,
          }));
          setPlayerData(mapped);
        }
      } catch (err) {
        console.error('Unexpected error:', err);
        setPlayerData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Filter by search term
  const filteredPlayers = playerData.filter((player) =>
    player.name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-xl font-medium">
        Loading projections...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#87CEEB] font-sans text-[#2c3e50]">
      <Navbar />

      <main className="mx-auto max-w-6xl px-6 pt-24 pb-12">
        {/* Header */}
        <div className="mb-8 text-center text-white">
          <h1 className="text-4xl font-bold drop-shadow-md">
            BOXSCORE BOT PROJECTIONS
          </h1>
          <p className="mt-2 text-lg font-medium text-white/90">
            Live Machine Learning Projections for {playerData[0]?.date || 'Upcoming Games'}
          </p>
        </div>

        {/* Search Bar */}
        <div className="mb-8 flex justify-center">
          <input
            type="text"
            placeholder="Search by player name..."
            className="w-full max-w-md rounded-full border-2 border-white bg-white/95 px-6 py-3 shadow-lg outline-none transition-all focus:border-[#8B0000] focus:ring-2 focus:ring-[#8B0000]/20"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Table */}
        <div className="overflow-hidden rounded-xl bg-white shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead className="bg-[#8B0000] text-white">
                <tr>
                  <th className="p-4 font-bold uppercase tracking-wider">Player</th>
                  <th className="p-4 font-bold uppercase tracking-wider">Team</th>
                  <th className="p-4 font-bold uppercase tracking-wider">Opp</th>
                  <th className="p-4 font-bold uppercase tracking-wider text-center">PTS</th>
                  <th className="p-4 font-bold uppercase tracking-wider text-center">REB</th>
                  <th className="p-4 font-bold uppercase tracking-wider text-center">AST</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-blue-50">
                {filteredPlayers.length > 0 ? (
                  filteredPlayers.map((player, index) => (
                    <tr key={index} className="hover:bg-blue-50 transition-colors">
                      <td className="p-4 font-semibold text-blue-900">{player.name}</td>
                      <td className="p-4 text-sm">{player.team}</td>
                      <td className="p-4 text-sm text-gray-500">@{player.opponent}</td>
                      <td className="p-4 text-center font-medium">{(player.points ?? 0).toFixed(2)}</td>
                      <td className="p-4 text-center font-medium">{(player.rebounds ?? 0).toFixed(2)}</td>
                      <td className="p-4 text-center font-medium">{(player.assists ?? 0).toFixed(2)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="p-12 text-center text-gray-400 italic">
                      No matching player found for "{searchTerm}"
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}