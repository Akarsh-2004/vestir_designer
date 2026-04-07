import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <section className="card">
      <h2>Oops—this page wandered off.</h2>
      <p className="muted" style={{ margin: '0 0 14px' }}>
        No worries. Let’s get you back to your wardrobe.
      </p>
      <Link className="btn" to="/">
        Back to Wardrobe
      </Link>
    </section>
  )
}
