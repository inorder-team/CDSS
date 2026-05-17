/**
 * CDSS Platform – API Client Utilities
 * Shared JavaScript functions for API communication.
 */

const CDSS_API_BASE = '/api/v1';

/**
 * Generic API request helper with error handling.
 */
async function apiRequest(method, path, body = null, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const options = { method, headers };
  if (body) options.body = JSON.stringify(body);

  const response = await fetch(`${CDSS_API_BASE}${path}`, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  }
  return data;
}

/**
 * Login and get token pair.
 */
async function login(username, password) {
  return await apiRequest('POST', '/auth/token', { username, password });
}

/**
 * Post CDSS recommendation request.
 */
async function getCDSSRecommendation(payload, token = null) {
  return await apiRequest('POST', '/recommendations', payload, token);
}

/**
 * Submit human review decision.
 */
async function submitReview(correlationId, action, reviewerId, notes = null, token = null) {
  return await apiRequest('POST', `/recommendations/${correlationId}/review`, {
    correlation_id: correlationId,
    reviewer_id: reviewerId,
    reviewer_role: 'CARDIOLOGIST',
    action,
    notes,
  }, token);
}

/**
 * Get medication information.
 */
async function getMedicationInfo(drugName) {
  return await apiRequest('GET', `/medications/${drugName}`);
}

/**
 * Check drug interactions.
 */
async function checkInteractions(drugs) {
  return await apiRequest('POST', '/medications/interactions', drugs);
}

/**
 * Get system health.
 */
async function getHealth() {
  return await apiRequest('GET', '/health');
}

/**
 * Format confidence score as percentage string.
 */
function formatConfidence(score) {
  return `${(score * 100).toFixed(0)}%`;
}

/**
 * Format ISO timestamp to human-readable.
 */
function formatTimestamp(iso) {
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
