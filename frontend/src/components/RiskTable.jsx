import { useEffect, useState } from 'react'
import { fetchAtRiskUsers } from '../api/client'

function scoreBadgeClass(score) {
  if (score > 85) return 'bg-red-600 text-white'
  if (score > 70) return 'bg-orange-500 text-white'
  if (score >= 40) return 'bg-yellow-500 text-black'
  return 'bg-green-500 text-white'
}

function SkeletonRow() {
  return (
    <tr className="animate-pulse">
      {[...Array(4)].map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 bg-gray-200 rounded w-full" />
        </td>
      ))}
    </tr>
  )
}

export default function RiskTable({ onSelectUser }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchAtRiskUsers(0)
      .then(data => {
        const sorted = [...(Array.isArray(data) ? data : [])].sort(
          (a, b) => b.risk_score - a.risk_score
        )
        setUsers(sorted)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="overflow-auto h-full">
      <table className="min-w-full text-sm border-collapse">
        <thead className="bg-gray-100 sticky top-0">
          <tr>
            <th className="px-4 py-2 text-left font-semibold text-gray-700">User ID</th>
            <th className="px-4 py-2 text-left font-semibold text-gray-700">Risk Score</th>
            <th className="px-4 py-2 text-left font-semibold text-gray-700">Tier</th>
            <th className="px-4 py-2 text-left font-semibold text-gray-700">Reason</th>
          </tr>
        </thead>
        <tbody>
          {loading && [...Array(8)].map((_, i) => <SkeletonRow key={i} />)}
          {!loading && error && (
            <tr>
              <td colSpan={4} className="px-4 py-6 text-center text-red-500">{error}</td>
            </tr>
          )}
          {!loading && !error && users.length === 0 && (
            <tr>
              <td colSpan={4} className="px-4 py-6 text-center text-gray-500">No users found.</td>
            </tr>
          )}
          {!loading && !error && users.map(user => (
            <tr
              key={user.user_id}
              className="border-b hover:bg-blue-50 cursor-pointer transition-colors"
              onClick={() => onSelectUser && onSelectUser(user)}
            >
              <td className="px-4 py-3 font-mono text-gray-800">
                {String(user.user_id).slice(0, 8)}
              </td>
              <td className="px-4 py-3">
                <span className={`inline-block px-2 py-1 rounded-full text-xs font-bold ${scoreBadgeClass(user.risk_score)}`}>
                  {user.risk_score}
                </span>
              </td>
              <td className="px-4 py-3 capitalize text-gray-700">{user.risk_tier}</td>
              <td className="px-4 py-3 text-gray-600 max-w-xs truncate">{user.reason ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
