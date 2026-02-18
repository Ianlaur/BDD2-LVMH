import { NavLink } from "react-router";
import { BarChart3, Clock, Users, Grid3x3 } from "lucide-react";

const navItems = [
  { path: "/", label: "Data", icon: Grid3x3 },
  { path: "/segments", label: "Segments", icon: BarChart3 },
  { path: "/clients", label: "Clients", icon: Users },
  { path: "/actions", label: "Actions", icon: Clock },
];

export function Sidebar() {
  return (
    <aside className="w-40 bg-white border-r border-neutral-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-neutral-200">
        <div className="mb-1">
          <div className="text-base font-semibold tracking-tight text-neutral-900">LVMH</div>
          <div className="text-xs text-neutral-500 font-normal">Client Intelligence</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                isActive
                  ? "bg-neutral-900 text-white"
                  : "text-neutral-700 hover:bg-neutral-100"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <item.icon className="size-4 flex-shrink-0" />
                <span className="text-sm font-medium">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-neutral-200 text-xs text-neutral-600">
        <div className="font-medium">400<span className="text-neutral-400 ml-0.5">CLIENTS</span></div>
        <div className="font-medium">8<span className="text-neutral-400 ml-0.5">SEGMENTS</span></div>
      </div>
    </aside>
  );
}