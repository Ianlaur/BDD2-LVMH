import { useState } from "react";
import { AlertCircle, Calendar, Tag, Users } from "lucide-react";

const actionTypes = [
  { label: "Toutes", value: "all", count: 0 },
  { label: "Urgentes", value: "urgent", count: 0 },
  { label: "Moyennes", value: "medium", count: 0 },
];

const statsData = [
  {
    label: "Urgentes",
    value: "0",
    icon: AlertCircle,
    color: "bg-neutral-100 text-neutral-900",
    borderColor: "border-neutral-200",
  },
  {
    label: "Visites",
    value: "0",
    icon: Calendar,
    color: "bg-neutral-100 text-neutral-900",
    borderColor: "border-neutral-200",
  },
  {
    label: "Ventes",
    value: "0",
    icon: Tag,
    color: "bg-neutral-100 text-neutral-900",
    borderColor: "border-neutral-200",
  },
  {
    label: "VIP",
    value: "0",
    icon: Users,
    color: "bg-neutral-100 text-neutral-900",
    borderColor: "border-neutral-200",
  },
];

export function ActionsPage() {
  const [activeFilter, setActiveFilter] = useState("all");

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b border-neutral-200 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-neutral-900">Actions Recommandées</h1>
            <p className="text-sm text-neutral-500 mt-1">Priorisez vos interactions client</p>
          </div>
          
          {/* Filters */}
          <div className="flex gap-2">
            {actionTypes.map((type) => (
              <button
                key={type.value}
                onClick={() => setActiveFilter(type.value)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeFilter === type.value
                    ? "bg-neutral-900 text-white shadow-sm"
                    : "bg-white text-neutral-700 border border-neutral-300 hover:bg-neutral-50"
                }`}
              >
                {type.label} ({type.count})
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {statsData.map((stat) => (
              <div
                key={stat.label}
                className={`bg-white rounded-xl shadow-sm border ${stat.borderColor} p-6 hover:shadow-md transition-shadow`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-4xl font-semibold text-neutral-900 mb-1">
                      {stat.value}
                    </div>
                    <div className="text-sm text-neutral-600">{stat.label}</div>
                  </div>
                  <div className={`p-3 rounded-lg ${stat.color}`}>
                    <stat.icon className="size-6" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Empty State */}
          <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-12">
            <div className="text-center max-w-md mx-auto">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-neutral-100 mb-4">
                <Calendar className="size-8 text-neutral-400" />
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                Aucune action en cours
              </h3>
              <p className="text-sm text-neutral-500 mb-6">
                Les actions recommandées apparaîtront ici en fonction de l'analyse de vos segments clients et de leurs comportements.
              </p>
              <div className="flex gap-3 justify-center">
                <button className="px-4 py-2 bg-neutral-900 text-white rounded-lg text-sm font-medium hover:bg-neutral-800 transition-colors">
                  Analyser les segments
                </button>
                <button className="px-4 py-2 bg-white border border-neutral-300 text-neutral-700 rounded-lg text-sm font-medium hover:bg-neutral-50 transition-colors">
                  Paramètres
                </button>
              </div>
            </div>
          </div>

          {/* Info Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-5">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-neutral-100 rounded-lg">
                  <Calendar className="size-5 text-neutral-900" />
                </div>
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-neutral-900 mb-1">
                    Planification automatique
                  </h4>
                  <p className="text-xs text-neutral-500">
                    Les actions sont générées automatiquement selon les priorités détectées
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-5">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-neutral-100 rounded-lg">
                  <Users className="size-5 text-neutral-900" />
                </div>
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-neutral-900 mb-1">
                    Segmentation intelligente
                  </h4>
                  <p className="text-xs text-neutral-500">
                    Basée sur 8 segments clients identifiés par clustering sémantique
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-5">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-neutral-100 rounded-lg">
                  <Tag className="size-5 text-neutral-900" />
                </div>
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-neutral-900 mb-1">
                    Opportunités ciblées
                  </h4>
                  <p className="text-xs text-neutral-500">
                    Recommandations personnalisées pour maximiser les conversions
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}