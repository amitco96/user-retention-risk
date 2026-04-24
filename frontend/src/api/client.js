import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchAtRiskUsers = (threshold = 0) =>
  api.get(`/users/at-risk?threshold=${threshold}`).then(r => r.data)

export const fetchUserRisk = (userId) =>
  api.get(`/users/${encodeURIComponent(userId)}/risk`).then(r => r.data)

export const postFeedback = (userId, feedback) =>
  api.post(`/users/${encodeURIComponent(userId)}/risk/feedback`, feedback).then(r => r.data)

export const fetchCohortRetention = () =>
  api.get('/cohorts/retention').then(r => r.data)
