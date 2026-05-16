import { Route, Routes } from 'react-router-dom'
import { Toaster } from 'sonner'
import Index from './pages/Index'
import NotFound from './pages/NotFound'
import { ItemDetailScreen } from './features/items/ItemDetailScreen'
import { FinishMyFitScreen } from './features/finish-my-fit/FinishMyFitScreen'
import { OutfitSuggestionsScreen } from './features/outfit-suggestions/OutfitSuggestionsScreen'
import { SettingsScreen } from './features/settings/SettingsScreen'
import { SizePassportScreen } from './features/settings/SizePassportScreen'
import { DetectionReviewSheet } from './features/add-items/DetectionReviewSheet'
import { ImportCaptureScreen } from './features/add-items/ImportCaptureScreen'
import { useWardrobeStore } from './store/wardrobeStore'

function App() {
  const pendingDetection = useWardrobeStore((s) => s.pendingDetection)
  return (
    <div className="app-shell">
      <main className="content">
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/item/:id" element={<ItemDetailScreen />} />
          <Route path="/finish-my-fit/:anchorId" element={<FinishMyFitScreen />} />
          <Route path="/outfit-suggestions/:anchorId" element={<OutfitSuggestionsScreen />} />
          <Route path="/settings" element={<SettingsScreen />} />
          <Route path="/settings/size-passport" element={<SizePassportScreen />} />
          <Route path="/import-capture/:token" element={<ImportCaptureScreen />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Toaster
        position="bottom-center"
        richColors
        closeButton
        toastOptions={{
          classNames: {
            toast: 'vestir-toast',
            title: 'vestir-toast__title',
            description: 'vestir-toast__description',
            closeButton: 'vestir-toast__close',
          },
        }}
      />
      {pendingDetection && <DetectionReviewSheet />}
    </div>
  )
}

export default App
