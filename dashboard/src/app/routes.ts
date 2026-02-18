import { createBrowserRouter } from "react-router";
import { RootLayout } from "./components/RootLayout";
import { DataPage } from "./pages/DataPage";
import { SegmentsPage } from "./pages/SegmentsPage";
import { ClientsPage } from "./pages/ClientsPage";
import { ActionsPage } from "./pages/ActionsPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      { index: true, Component: DataPage },
      { path: "segments", Component: SegmentsPage },
      { path: "clients", Component: ClientsPage },
      { path: "actions", Component: ActionsPage },
    ],
  },
]);
