import { useState } from 'react'
import RiskTable from './components/RiskTable'
import UserRiskCard from './components/UserRiskCard'
import CohortChart from './components/CohortChart'

export default function App() {
  const [selectedUser, setSelectedUser] = useState(null)

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <h1 className="text-xl font-bold text-gray-900 tracking-tight">
          User Retention Risk Dashboard
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">Real-time churn risk scoring powered by XGBoost + Claude</p>
      </header>

      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex flex-1 overflow-hidden" style={{ minHeight: '420px' }}>
          <section className="w-3/5 bg-white border-r border-gray-200 overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">At-Risk Users</h2>
            </div>
            <div className="flex-1 overflow-auto">
              <RiskTable onSelectUser={u => setSelectedUser(u.user_id)} />
            </div>
          </section>

          <section className="w-2/5 bg-white overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">User Detail</h2>
            </div>
            <div className="flex-1 overflow-auto">
              <UserRiskCard userId={selectedUser} />
            </div>
          </section>
        </div>

        <section className="bg-white border-t border-gray-200" style={{ minHeight: '300px' }}>
          <CohortChart />
        </section>
      </main>
    </div>
  )
}
