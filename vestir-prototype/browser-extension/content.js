function isVisibleImage(img) {
  const rect = img.getBoundingClientRect()
  const style = window.getComputedStyle(img)
  if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false
  if (rect.width < 140 || rect.height < 140) return false
  return true
}

function scoreImage(img) {
  const rect = img.getBoundingClientRect()
  const area = rect.width * rect.height
  const nearViewportTop = rect.top >= -200 && rect.top < window.innerHeight * 1.5
  return area + (nearViewportTop ? 100000 : 0)
}

function pickBestProductImage() {
  const images = [...document.images]
    .filter((img) => Boolean(img.currentSrc || img.src))
    .filter(isVisibleImage)
    .sort((a, b) => scoreImage(b) - scoreImage(a))
  const best = images[0]
  if (!best) return null
  return best.currentSrc || best.src || null
}

function pickTitle() {
  const og = document.querySelector('meta[property="og:title"]')?.getAttribute('content')
  if (og && og.trim()) return og.trim()
  const h1 = document.querySelector('h1')?.textContent
  if (h1 && h1.trim()) return h1.trim()
  return document.title || null
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'VESTIR_EXTRACT_PRODUCT') return
  sendResponse({
    ok: true,
    pageUrl: window.location.href,
    title: pickTitle(),
    imageUrl: pickBestProductImage(),
  })
})
