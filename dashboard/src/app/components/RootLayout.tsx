import { Outlet } from "react-router";
import { Sidebar } from "./Sidebar";

export function RootLayout() {
  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <div className="flex-1">
        <Outlet />
      </div>
    </div>
  );
}