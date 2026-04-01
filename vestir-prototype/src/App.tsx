import { Route, Routes } from 'react-router-dom'
import { Toaster } from 'sonner'
import { NavLink } from './components/NavLink'
import Index from './pages/Index'
import NotFound from './pages/NotFound'
import { ItemDetailScreen } from './features/items/ItemDetailScreen'
import { FinishMyFitScreen } from './features/finish-my-fit/FinishMyFitScreen'
import { SettingsScreen } from './features/settings/SettingsScreen'
import { DetectionReviewSheet } from './features/add-items/DetectionReviewSheet'
import { useWardrobeStore } from './store/wardrobeStore'

function App() {
  const pendingDetection = useWardrobeStore((s) => s.pendingDetection)
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">VESTIR Prototype</div>
        <nav className="nav">
          <NavLink to="/">Home</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
      </header>

      <main className="content">
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/item/:id" element={<ItemDetailScreen />} />
          <Route path="/finish-my-fit/:anchorId" element={<FinishMyFitScreen />} />
          <Route path="/settings" element={<SettingsScreen />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Toaster position="top-center" />
      {pendingDetection && <DetectionReviewSheet />}
    </div>
  )
}

export default App
