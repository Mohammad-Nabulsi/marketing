import React from "react";

type Props = {
  title?: string;
  rows: Record<string, any>[];
  maxRows?: number;
};

function stringify(v: any) {
  if (v === null || v === undefined) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export default function DataTable({ title, rows, maxRows = 50 }: Props) {
  const sliced = rows.slice(0, maxRows);
  const cols = React.useMemo(() => {
    const set = new Set<string>();
    sliced.forEach((r) => Object.keys(r || {}).forEach((k) => set.add(k)));
    return Array.from(set);
  }, [sliced]);

  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/40">
      {title ? <div className="border-b border-slate-800 px-4 py-3 text-sm font-semibold">{title}</div> : null}
      {sliced.length === 0 ? (
        <div className="px-4 py-6 text-sm text-slate-400">No data.</div>
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="sticky top-0 bg-slate-950">
              <tr>
                {cols.map((c) => (
                  <th key={c} className="whitespace-nowrap border-b border-slate-800 px-3 py-2 text-xs font-semibold text-slate-300">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sliced.map((r, i) => (
                <tr key={i} className="odd:bg-slate-900/20">
                  {cols.map((c) => (
                    <td key={c} className="max-w-[360px] truncate border-b border-slate-900 px-3 py-2 text-slate-200" title={stringify(r?.[c])}>
                      {stringify(r?.[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {rows.length > maxRows ? <div className="px-4 py-2 text-xs text-slate-500">Showing first {maxRows} rows.</div> : null}
    </div>
  );
}

