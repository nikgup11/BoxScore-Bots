import Navbar from '@/components/nav_bar';

export default function StatsPage() {
  // Hard-coded values for now
  const playerData = [
    { name: "LeBron James", points: 25.7, rebounds: 7.3, assists: 8.3 },
    { name: "Stephen Curry", points: 26.4, rebounds: 4.5, assists: 5.1 },
    { name: "Nikola Jokic", points: 26.1, rebounds: 12.4, assists: 9.0 },
    { name: "Luka Doncic", points: 33.9, rebounds: 9.2, assists: 9.8 },
  ];

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
              {playerData.map((player, index) => (
                <tr 
                  key={index} 
                  className="hover:bg-blue-50 transition-colors duration-150"
                >
                  <td className="p-4 font-semibold text-blue-900">{player.name}</td>
                  <td className="p-4">{player.points}</td>
                  <td className="p-4">{player.rebounds}</td>
                  <td className="p-4">{player.assists}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
