import { PieChart, Pie, Cell, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend } from "recharts";

// Palette de 8 couleurs vives, une par segment (UI principale reste en noir & blanc)
const segmentsData = [
  {
    id: 0,
    name: "Segment 0",
    clients: 63,
    color: "#1d4ed8", // bleu
    description: "Collectionneur d'Art | Philanthrope | Sac Cuir Exotique",
  },
  {
    id: 1,
    name: "Segment 1",
    clients: 48,
    color: "#16a34a", // vert
    description: "PDG | Visite Privée Demandée | Haute Joaillerie",
  },
  {
    id: 2,
    name: "Segment 2",
    clients: 39,
    color: "#f97316", // orange
    description: "Champagne Millésimé Rare | Prêt à Acheter | Visite Privée Demandée",
  },
  {
    id: 3,
    name: "Segment 3",
    clients: 58,
    color: "#dc2626", // rouge
    description: "Visite Privée Demandée | Client VIP | Haute Couture",
  },
  {
    id: 4,
    name: "Segment 4",
    clients: 34,
    color: "#7c3aed", // violet
    description: "Mécénat Immobilier | Sac Cuir Exotique | Visite Privée Demandée",
  },
  {
    id: 5,
    name: "Segment 5",
    clients: 45,
    color: "#0891b2", // cyan
    description: "Édition Limitée | Client VIP | Capital-risqueur",
  },
  {
    id: 6,
    name: "Segment 6",
    clients: 41,
    color: "#eab308", // jaune
    description: "Fondateur Tech | Malle Sur Mesure | Visite Privée Demandée",
  },
  {
    id: 7,
    name: "Segment 7",
    clients: 74,
    color: "#db2777", // rose
    description: "Client VIP | Capital-risqueur | Sac Cuir Exotique",
  },
];

const radarData = [
  { subject: "Loisirs", "Seg 0": 85, "Seg 1": 70, "Seg 2": 60, "Seg 3": 75 },
  { subject: "Voyage", "Seg 0": 70, "Seg 1": 85, "Seg 2": 55, "Seg 3": 65 },
  { subject: "Mode", "Seg 0": 90, "Seg 1": 80, "Seg 2": 75, "Seg 3": 85 },
  { subject: "Famille", "Seg 0": 60, "Seg 1": 65, "Seg 2": 70, "Seg 3": 80 },
  { subject: "Budget", "Seg 0": 95, "Seg 1": 90, "Seg 2": 70, "Seg 3": 85 },
  { subject: "VIP", "Seg 0": 80, "Seg 1": 75, "Seg 2": 60, "Seg 3": 90 },
  { subject: "Contraintes", "Seg 0": 40, "Seg 1": 50, "Seg 2": 65, "Seg 3": 45 },
  { subject: "Cadeaux", "Seg 0": 75, "Seg 1": 70, "Seg 2": 80, "Seg 3": 70 },
];

export function SegmentsPage() {
  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b border-neutral-200 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-900">Analyse des Segments</h1>
        <p className="text-sm text-neutral-500 mt-1">8 segments par clustering sémantique</p>
      </header>

      <div className="p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Segments Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {segmentsData.map((segment) => (
              <div
                key={segment.id}
                className="bg-white rounded-xl shadow-sm border border-neutral-200 p-5 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start gap-3 mb-3">
                  <div
                    className="w-12 h-12 rounded-lg flex items-center justify-center text-white font-semibold flex-shrink-0"
                    style={{ backgroundColor: segment.color }}
                  >
                    {segment.id}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-neutral-900 mb-0.5">
                      {segment.clients} clients
                    </div>
                    <div className="text-xs text-neutral-500">{segment.name}</div>
                  </div>
                </div>
                <div className="text-xs text-neutral-600 leading-relaxed line-clamp-3">
                  {segment.description}
                </div>
                <div className="mt-3 pt-3 border-t border-neutral-100">
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      backgroundColor: segment.color,
                      width: `${(segment.clients / 74) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Distribution Chart */}
            <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6">
              <h2 className="text-lg font-semibold text-neutral-900 mb-6">Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={segmentsData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={2}
                    dataKey="clients"
                  >
                    {segmentsData.map((entry) => (
                      <Cell key={`cell-${entry.id}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-x-4 gap-y-2 mt-4 justify-center">
                {segmentsData.map((segment) => (
                  <div key={segment.id} className="flex items-center gap-1.5">
                    <div
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: segment.color }}
                    />
                    <span className="text-xs text-neutral-600">{segment.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Radar Chart */}
            <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6">
              <h2 className="text-lg font-semibold text-neutral-900 mb-6">Profil Radar</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#e5e5e5" />
                  <PolarAngleAxis 
                    dataKey="subject" 
                    tick={{ fill: '#737373', fontSize: 11 }}
                  />
                  <Radar
                    name="Seg 0"
                    dataKey="Seg 0"
                    stroke={segmentsData[0].color}
                    fill={segmentsData[0].color}
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="Seg 1"
                    dataKey="Seg 1"
                    stroke={segmentsData[1].color}
                    fill={segmentsData[1].color}
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="Seg 2"
                    dataKey="Seg 2"
                    stroke={segmentsData[2].color}
                    fill={segmentsData[2].color}
                    fillOpacity={0.2}
                  />
                  <Radar
                    name="Seg 3"
                    dataKey="Seg 3"
                    stroke={segmentsData[3].color}
                    fill={segmentsData[3].color}
                    fillOpacity={0.2}
                  />
                </RadarChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-x-4 gap-y-2 mt-4 justify-center">
                {segmentsData.slice(0, 4).map((seg) => (
                  <div key={seg.name} className="flex items-center gap-1.5">
                    <div
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: seg.color }}
                    />
                    <span className="text-xs text-neutral-600">{seg.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}