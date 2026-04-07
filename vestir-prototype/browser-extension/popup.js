const apiInput = document.getElementById('apiBase')
const matchBtn = document.getElementById('matchBtn')
const statusEl = document.getElementById('status')
const matchesEl = document.getElementById('matches')
const queryCardEl = document.getElementById('queryCard')
const queryImageEl = document.getElementById('queryImage')

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true })
  return tabs[0]
}

function setStatus(text, isError = false) {
  statusEl.textContent = text
  statusEl.style.color = isError ? '#B91C1C' : '#374151'
}

function resolveImageUrl(url, apiBase) {
  if (!url) return ''
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) return url
  if (url.startsWith('/')) return `${apiBase}${url}`
  return `${apiBase}/${url}`
}

function renderMatches(data, apiBase) {
  matchesEl.innerHTML = ''
  const queryImageUrl = resolveImageUrl(data?.query?.processed_image_url || data?.query?.image_url, apiBase)
  if (queryImageUrl) {
    queryCardEl.hidden = false
    queryImageEl.src = queryImageUrl
  } else {
    queryCardEl.hidden = true
    queryImageEl.removeAttribute('src')
  }

  const list = Array.isArray(data.matches) ? data.matches : []
  if (!list.length) {
    const li = document.createElement('li')
    li.className = 'match-info'
    li.textContent = 'No closet matches found yet.'
    matchesEl.appendChild(li)
    return
  }
  for (const match of list) {
    const li = document.createElement('li')
    const image = document.createElement('img')
    image.className = 'match-image'
    const label = match.snapshot
      ? `${match.snapshot.color_primary} ${match.snapshot.item_type} (${match.snapshot.category})`
      : `Item ${match.item_id}`
    const imageUrl = resolveImageUrl(match?.snapshot?.image_url, apiBase)
    if (imageUrl) {
      image.src = imageUrl
      image.alt = label
      li.appendChild(image)
    }

    const info = document.createElement('div')
    info.className = 'match-info'
    const title = document.createElement('div')
    title.className = 'match-title'
    title.textContent = label
    const score = document.createElement('div')
    score.textContent = `Similarity ${(match.score * 100).toFixed(1)}%`
    info.appendChild(title)
    info.appendChild(score)
    li.appendChild(info)
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
    renderMatches(data, apiBase)
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
