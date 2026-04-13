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

function pickOne(arr) {
  return arr[Math.floor(Math.random() * arr.length)]
}

async function collectImages(dir, out) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      await collectImages(p, out)
      continue
    }
    if (/\.(jpg|jpeg|png)$/i.test(entry.name)) out.push(p)
  }
}

async function main() {
  const explicitIds = process.argv.slice(2)
  const roots = [
    path.join(workspaceRoot, 'DeepFashion2'),
    path.join(workspaceRoot, 'Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion', 'data', 'samples'),
  ]
  const images = []
  for (const root of roots) {
    try {
      await fs.access(root)
      await collectImages(root, images)
    } catch {
      // ignore
    }
  }
  if (!images.length) throw new Error('No source images found for snapshot backfill.')
  await fs.mkdir(processedDir, { recursive: true })
  const entries = JSON.parse(await fs.readFile(embeddingsFile, 'utf8'))
  let imageIdx = 0
  let touched = 0
  for (const entry of entries) {
    const shouldTarget = explicitIds.length ? explicitIds.includes(entry.item_id) : true
    if (!shouldTarget) continue
    if (entry.item_snapshot?.image_url) continue
    const src = images[imageIdx % images.length]
    imageIdx += 1
    const ext = path.extname(src) || '.jpg'
    const filename = `${crypto.randomUUID()}${ext}`
    await fs.copyFile(src, path.join(processedDir, filename))
    const category = pickOne(categories)
    entry.item_snapshot = {
      item_id: entry.item_id,
      image_url: `/storage/processed/${filename}`,
      item_type: pickOne(typesByCategory[category]),
      category,
      color_primary: pickOne(colors),
      material: pickOne(materials),
    }
    touched += 1
  }
  await fs.writeFile(embeddingsFile, JSON.stringify(entries), 'utf8')
  console.log(`Backfilled snapshots for ${touched} entries.`)
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err))
  process.exit(1)
})
