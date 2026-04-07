import fs from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const workspaceRoot = path.resolve(projectRoot, '..')
const processedDir = path.join(projectRoot, 'server', 'storage', 'processed')
const embeddingsFile = path.join(projectRoot, 'server', 'storage', 'embeddings.json')

const deepfashionRoots = [
  path.join(workspaceRoot, 'DeepFashion2'),
  path.join(workspaceRoot, 'Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion', 'data', 'samples'),
]

const categories = ['Tops', 'Bottoms', 'Outerwear', 'Shoes', 'Accessories']
const colors = ['Black', 'White', 'Gray', 'Navy', 'Olive', 'Taupe', 'Cream']
const materials = ['Cotton', 'Denim', 'Wool', 'Polyester', 'Linen']
const typesByCategory = {
  Tops: ['Shirt', 'Tee', 'Blouse', 'Sweater'],
  Bottoms: ['Jeans', 'Trousers', 'Shorts', 'Skirt'],
  Outerwear: ['Jacket', 'Coat', 'Hoodie', 'Blazer'],
  Shoes: ['Sneakers', 'Boots', 'Loafers'],
  Accessories: ['Bag', 'Belt', 'Scarf', 'Cap'],
}

function seededVector(seedText, dims = 3072) {
  const hash = crypto.createHash('sha256').update(seedText).digest('hex')
  let x = Number.parseInt(hash.slice(0, 8), 16) || 7
  const values = []
  for (let i = 0; i < dims; i += 1) {
    x = (x * 48271) % 2147483647
    values.push((x / 2147483647) * 2 - 1)
  }
  return values
}

async function walkImages(dir, out) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      await walkImages(fullPath, out)
      continue
    }
    if (/\.(jpg|jpeg|png)$/i.test(entry.name)) out.push(fullPath)
  }
}

async function collectImages() {
  const all = []
  for (const root of deepfashionRoots) {
    try {
      await fs.access(root)
      await walkImages(root, all)
    } catch {
      // ignore missing roots
    }
  }
  return all
}

function pickRandom(items, count) {
  const arr = [...items]
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr.slice(0, count)
}

function pickOne(arr) {
  return arr[Math.floor(Math.random() * arr.length)]
}

async function main() {
  const countArg = Number.parseInt(process.argv[2] ?? '12', 10)
  const count = Number.isFinite(countArg) ? Math.max(1, Math.min(200, countArg)) : 12
  await fs.mkdir(processedDir, { recursive: true })
  const allImages = await collectImages()
  if (!allImages.length) {
    throw new Error('No DeepFashion images found. Expected DeepFashion2 or sidecar sample images in workspace.')
  }
  const selected = pickRandom(allImages, Math.min(count, allImages.length))
  const existing = JSON.parse(await fs.readFile(embeddingsFile, 'utf8'))
  for (const sourcePath of selected) {
    const ext = path.extname(sourcePath).toLowerCase() || '.jpg'
    const filename = `${crypto.randomUUID()}${ext}`
    await fs.copyFile(sourcePath, path.join(processedDir, filename))
    const category = pickOne(categories)
    const itemType = pickOne(typesByCategory[category])
    const color = pickOne(colors)
    const itemId = crypto.randomUUID()
    existing.push({
      id: crypto.randomUUID(),
      item_id: itemId,
      vector: seededVector(`${sourcePath}|${category}|${itemType}|${color}`),
      created_at: new Date().toISOString(),
      model: 'deepfashion-demo-seed-v1',
      item_snapshot: {
        item_id: itemId,
        image_url: `/storage/processed/${filename}`,
        item_type: itemType,
        category,
        color_primary: color,
        material: pickOne(materials),
      },
    })
  }
  await fs.writeFile(embeddingsFile, JSON.stringify(existing), 'utf8')
  console.log(`Seeded ${selected.length} DeepFashion demo closet items.`)
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
})
