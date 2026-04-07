const apiInput = document.getElementById('apiBase')
const matchBtn = document.getElementById('matchBtn')
const statusEl = document.getElementById('status')
const matchesEl = document.getElementById('matches')

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true })
  return tabs[0]
}

function setStatus(text, isError = false) {
  statusEl.textContent = text
  statusEl.style.color = isError ? '#B91C1C' : '#374151'
}

function renderMatches(data) {
  matchesEl.innerHTML = ''
  const list = Array.isArray(data.matches) ? data.matches : []
  if (!list.length) {
    const li = document.createElement('li')
    li.textContent = 'No closet matches found yet.'
    matchesEl.appendChild(li)
    return
  }
  for (const match of list) {
    const li = document.createElement('li')
    const label = match.snapshot
      ? `${match.snapshot.color_primary} ${match.snapshot.item_type} (${match.snapshot.category})`
      : `Item ${match.item_id}`
    li.textContent = `${label} - ${(match.score * 100).toFixed(1)}%`
    matchesEl.appendChild(li)
  }
}

async function runMatch() {
  try {
    setStatus('Reading product from current page...')
    const tab = await getActiveTab()
    if (!tab?.id) throw new Error('No active tab found')
    const extraction = await chrome.tabs.sendMessage(tab.id, { type: 'VESTIR_EXTRACT_PRODUCT' })
    if (!extraction?.imageUrl) {
      throw new Error('Could not find a clothing image on this page.')
    }
    const apiBase = (apiInput.value || 'http://127.0.0.1:8787').replace(/\/$/, '')
    await chrome.storage.local.set({ vestirApiBase: apiBase })

    setStatus('Matching against closet...')
    const response = await fetch(`${apiBase}/api/extension/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        imageUrl: extraction.imageUrl,
        pageUrl: extraction.pageUrl,
        title: extraction.title,
        topK: 5,
      }),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error ?? 'Match request failed')
    renderMatches(data)
    setStatus('Done')
  } catch (error) {
    setStatus(error instanceof Error ? error.message : 'Unexpected error', true)
  }
}

async function init() {
  const saved = await chrome.storage.local.get(['vestirApiBase'])
  apiInput.value = saved.vestirApiBase || 'http://127.0.0.1:8787'
  matchBtn.addEventListener('click', runMatch)
}

init()
