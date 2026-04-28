import { NavLink, Outlet } from "react-router-dom";
import React from "react";

const navItems = [
  { to: "/", label: "Upload & Data Quality" },
  { to: "/kpis", label: "KPI Dashboard" },
  { to: "/content", label: "Content Performance" },
  { to: "/clustering", label: "Clustering & PCA" },
  { to: "/rules", label: "Association Rules" },
  { to: "/trends", label: "Trends & Forecasting" },
  { to: "/network", label: "Network Analysis" },
  { to: "/recommendations", label: "Recommendations" }
];

export function useDatasetId() {
  const [datasetId, setDatasetId] = React.useState<string>(() => localStorage.getItem("dataset_id") || "");
  React.useEffect(() => {
    localStorage.setItem("dataset_id", datasetId);
  }, [datasetId]);
  return { datasetId, setDatasetId };
}

export default function Layout() {
  const { datasetId, setDatasetId } = useDatasetId();

  return (
    <div className="h-full">
      <div className="flex h-full">
        <aside className="w-72 shrink-0 border-r border-slate-800 bg-slate-950">
          <div className="px-4 py-4">
            <div className="text-sm font-semibold leading-tight">
              Palestine SME
              <div className="text-xs font-normal text-slate-400">Social Media Intelligence</div>
            </div>
            <div className="mt-4">
              <label className="text-xs text-slate-400">Dataset ID</label>
              <input
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                placeholder="Upload to generate…"
                className="mt-1 w-full rounded-md border border-slate-800 bg-slate-900/40 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 outline-none focus:border-slate-600"
              />
              <div className="mt-1 text-xs text-slate-500">Stored in localStorage.</div>
            </div>
          </div>
          <nav className="px-2 pb-4">
            {navItems.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  [
                    "block rounded-md px-3 py-2 text-sm",
                    isActive ? "bg-slate-900 text-slate-100" : "text-slate-300 hover:bg-slate-900/50"
                  ].join(" ")
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className="flex-1 overflow-auto">
          <div className="mx-auto w-full max-w-6xl px-6 py-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

