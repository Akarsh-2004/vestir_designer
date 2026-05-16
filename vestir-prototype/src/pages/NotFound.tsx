import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <main className="notfound-screen">
      <section className="card" aria-labelledby="notfound-title">
        <h2 id="notfound-title" style={{ margin: '0 0 8px', fontSize: 20, fontWeight: 600, lineHeight: 1.25 }}>
          This page is not in your closet
        </h2>
        <p className="muted" style={{ margin: '0 0 18px', fontSize: 15, lineHeight: 1.45 }}>
          The link may be wrong or the page was removed. Head back to your wardrobe to keep browsing.
        </p>
        <Link className="btn btn-full" to="/">
          Back to wardrobe
        </Link>
      </section>
    </main>
  )
}
