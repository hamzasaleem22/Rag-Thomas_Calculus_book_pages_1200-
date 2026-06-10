import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './katex-fix.css'  // Loaded LAST to override Tailwind's border reset on KaTeX
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
