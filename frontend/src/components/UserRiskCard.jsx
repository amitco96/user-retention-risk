import { useEffect, useState } from 'react'
import { fetchUserRisk, postFeedback } from '../api/client'

function tierColor(tier) {
  const map = {
    low: 'text-green-500',
    medium: 'text-yellow-500',
    high: 'text-orange-500',
    critical: 'text-red-600',
  }
  return map[tier] ?? 'text-gray-600'
}

function driverPillClass(i) {
  const colors = [
    'bg-red-100 text-red-800',
    'bg-orange-100 text-orange-800',
    'bg-yellow-100 text-yellow-800',
  ]
  return colors[i] ?? 'bg-gray-100 text-gray-700'
}

function ScoreArc({ score }) {
  const radius = 60
  const circumference = Math.PI * radius
  const pct = Math.min(Math.max(score, 0), 100) / 100
  const dash = pct * circumference
  const gap = circumference - dash

  let strokeColor = '#22c55e'
  if (score > 85) strokeColor = '#dc2626'
  else if (score > 70) strokeColor = '#f97316'
  else if (score >= 40) strokeColor = '#eab308'

  return (
    <svg width="150" height="90" viewBox="0 0 150 90" className="mx-auto">
      <path
        d="M 15 78 A 60 60 0 0 1 135 78"
        fill="none"
        stroke="#e5e7eb"
        strokeWidth="12"
        strokeLinecap="round"
      />
      <path
        d="M 15 78 A 60 60 0 0 1 135 78"
        fill="none"
        stroke={strokeColor}
        strokeWidth="12"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${gap}`}
        style={{ transition: 'stroke-dasharray 0.5s ease' }}
      />
      <text x="75" y="72" textAnchor="middle" fontSize="30" fontWeight="bold" fill={strokeColor}>
        {score}
      </text>
      <text x="75" y="88" textAnchor="middle" fontSize="10" fill="#6b7280">/ 100</text>
    </svg>
  )
}

export default function UserRiskCard({ userId }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [actioned, setActioned] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!userId) return
    setLoading(true)
    setActioned(false)
    setError(null)
    fetchUserRisk(userId)
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [userId])

  const handleAction = () => {
    setActionLoading(true)
    postFeedback(userId, { feedback: 'actioned' })
      .then(() => setActioned(true))
      .catch(() => setActioned(true))
      .finally(() => setActionLoading(false))
  }

  if (!userId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Select a user from the table to view details.
      </div>
    )
  }

  if (loading) {
    return (
      <div className="p-6 space-y-4 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/2" />
        <div className="h-24 bg-gray-200 rounded" />
        <div className="h-4 bg-gray-200 rounded w-3/4" />
        <div className="h-4 bg-gray-200 rounded w-2/3" />
      </div>
    )
  }

  if (error) {
    return <div className="p-6 text-red-500 text-sm">{error}</div>
  }

  if (!data) return null

  const drivers = data.top_drivers ?? []

  return (
    <div className="p-6 space-y-5 overflow-auto h-full">
      <div>
        <p className="text-xs text-gray-500 font-mono mb-1">User {String(data.user_id).slice(0, 12)}</p>
        <p className={`text-lg font-bold capitalize ${tierColor(data.risk_tier)}`}>
          {data.risk_tier} Risk
        </p>
      </div>

      <ScoreArc score={data.risk_score} />

      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Top Risk Drivers</p>
        <div className="flex flex-wrap gap-2">
          {drivers.slice(0, 3).map((d, i) => (
            <span key={i} className={`px-3 py-1 rounded-full text-xs font-medium ${driverPillClass(i)}`}>
              {d}
            </span>
          ))}
          {drivers.length === 0 && <span className="text-gray-400 text-xs">No drivers available</span>}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-1">Why at risk</p>
        <p className="text-sm text-gray-800">{data.reason ?? 'No explanation available.'}</p>
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-1">Recommended action</p>
        <p className="text-sm text-gray-800 mb-3">{data.recommended_action ?? 'No action recommended.'}</p>
        {actioned ? (
          <p className="text-green-600 text-sm font-medium">Marked as actioned.</p>
        ) : (
          <button
            onClick={handleAction}
            disabled={actionLoading}
            className="bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors"
          >
            {actionLoading ? 'Saving...' : 'Mark as Actioned'}
          </button>
        )}
      </div>

      <p className="text-xs text-gray-400">
        Scored at: {data.scored_at ? new Date(data.scored_at).toLocaleString() : 'N/A'} &bull; Model: {data.model_version ?? 'N/A'}
      </p>
    </div>
  )
}
