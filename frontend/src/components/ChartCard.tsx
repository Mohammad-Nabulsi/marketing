import React from "react";

export default function ChartCard(props: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/40 p-4">
      <div className="text-sm font-semibold text-slate-100">{props.title}</div>
      <div className="mt-3 h-72 w-full">{props.children}</div>
    </div>
  );
}

