import { useState } from "react";
import { Search } from "lucide-react";

const segmentColors: Record<number, string> = {
  0: "#18181b",
  1: "#3f3f46",
  2: "#71717a",
  3: "#a1a1aa",
  4: "#525252",
  5: "#57534e",
  6: "#44403c",
  7: "#292524",
};

// Generate mock client data
const generateClients = () => {
  const clients = [];
  const prefixes = ["SA_", "SA_"];
  
  for (let i = 1; i <= 75; i++) {
    const id = i.toString().padStart(3, "0");
    const segment = Math.floor(Math.random() * 8);
    const similarity = 80 + Math.floor(Math.random() * 20);
    
    clients.push({
      id: `${prefixes[i % 2]}${id}${i > 50 ? i : ""}`,
      segment,
      similarity,
      color: segmentColors[segment],
    });
  }
  
  return clients;
};

const allClients = generateClients();

export function ClientsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<"all" | "segment">("all");

  const filteredClients = allClients.filter((client) =>
    client.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b border-neutral-200 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-neutral-900">Base Clients</h1>
            <p className="text-sm text-neutral-500 mt-1">400 clients analysés</p>
          </div>
          
          {/* Search */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-neutral-400" />
              <input
                type="text"
                placeholder="Rechercher..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-4 py-2 border border-neutral-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent w-64"
              />
            </div>
            
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as "all" | "segment")}
              className="px-4 py-2 border border-neutral-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent"
            >
              <option value="all">Tous</option>
              <option value="segment">Par segment</option>
            </select>
          </div>
        </div>
      </header>

      {/* Clients Grid */}
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredClients.map((client) => (
              <div
                key={client.id}
                className="bg-white rounded-xl shadow-sm border border-neutral-200 p-5 hover:shadow-md transition-all hover:-translate-y-0.5 cursor-pointer"
              >
                <div className="flex items-center justify-between mb-3">
                  <div
                    className="w-12 h-12 rounded-lg flex items-center justify-center text-white font-semibold text-sm"
                    style={{ backgroundColor: client.color }}
                  >
                    {client.segment}
                  </div>
                  
                  {/* Similarity badge */}
                  <div className="relative size-14">
                    <svg className="size-14 -rotate-90" viewBox="0 0 36 36">
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="#e5e5e5"
                        strokeWidth="2.5"
                      />
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke={client.color}
                        strokeWidth="2.5"
                        strokeDasharray={`${client.similarity}, 100`}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-xs font-semibold text-neutral-900">
                        {client.similarity}%
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-sm font-semibold text-neutral-900 mb-0.5">
                    {client.id}
                  </div>
                  <div className="text-xs text-neutral-500">Segment {client.segment}</div>
                </div>
              </div>
            ))}
          </div>

          {filteredClients.length === 0 && (
            <div className="text-center py-12">
              <p className="text-neutral-500">Aucun client trouvé</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}