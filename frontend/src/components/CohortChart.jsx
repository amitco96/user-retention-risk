import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { fetchCohortRetention } from '../api/client'

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6366f1',
]

function buildChartData(cohorts) {
  if (!cohorts || cohorts.length === 0) return { series: [], maxWeeks: 0 }

  const maxWeeks = Math.max(...cohorts.map(c => c.weeks?.length ?? 0))
  const series = []

  for (let w = 0; w < maxWeeks; w++) {
    const point = { week: w }
    cohorts.forEach(c => {
      const entry = c.weeks?.find(e => e.week === w)
      point[c.cohort_week] = entry ? entry.retention_pct : null
    })
    series.push(point)
  }

  return { series, maxWeeks }
}

export default function CohortChart() {
  const [cohorts, setCohorts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchCohortRetention()
      .then(data => setCohorts(data.cohorts ?? []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="animate-pulse p-6 space-y-3">
        <div className="h-4 bg-gray-200 rounded w-1/4" />
        <div className="h-48 bg-gray-200 rounded" />
      </div>
    )
  }

  if (error) {
    return <div className="p-6 text-red-500 text-sm">{error}</div>
  }

  if (cohorts.length === 0) {
    return <div className="p-6 text-gray-400 text-sm">No cohort data available.</div>
  }

  const { series } = buildChartData(cohorts)

  return (
    <div className="p-6 h-full">
      <h2 className="text-sm font-semibold text-gray-700 mb-4 uppercase tracking-wide">
        Cohort Retention (Week-over-Week)
      </h2>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={series} margin={{ top: 4, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="week"
            label={{ value: 'Weeks since signup', position: 'insideBottom', offset: -2, fontSize: 11 }}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={v => `${v}%`}
            tick={{ fontSize: 11 }}
          />
          <Tooltip formatter={(v) => v !== null ? `${Number(v).toFixed(1)}%` : 'N/A'} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {cohorts.map((c, i) => (
            <Line
              key={c.cohort_week}
              type="monotone"
              dataKey={c.cohort_week}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
