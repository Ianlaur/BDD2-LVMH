import { Vector3D } from "../components/Vector3D";
import { Users, Clock, Tag, Zap } from "lucide-react";

const stats = [
  { label: "Clients", value: "400", icon: Users, color: "bg-neutral-100 text-neutral-900" },
  { label: "Segments", value: "8", icon: Clock, color: "bg-neutral-100 text-neutral-900" },
  { label: "Concepts", value: "20", icon: Tag, color: "bg-neutral-100 text-neutral-900" },
  { label: "Moy/client", value: "3", icon: Zap, color: "bg-neutral-100 text-neutral-900" },
];

export function DataPage() {
  return (
    <div className="h-screen flex flex-col bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b border-neutral-200 px-8 py-6">
        <p className="text-sm text-neutral-500 mb-1">Cliquez sur les éléments pour explorer les données</p>
        <h1 className="text-2xl font-semibold text-neutral-900">Espace Vectoriel 3D</h1>
      </header>

      {/* Main Content */}
      <div className="flex-1 p-8 overflow-auto">
        <div className="max-w-7xl mx-auto">
          {/* Vector 3D Component */}
          <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6 mb-6">
            <Vector3D />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-3xl font-semibold text-neutral-900 mb-1">
                      {stat.value}
                    </div>
                    <div className="text-sm text-neutral-500">{stat.label}</div>
                  </div>
                  <div className={`p-3 rounded-lg ${stat.color}`}>
                    <stat.icon className="size-5" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}