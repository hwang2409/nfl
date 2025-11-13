import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NFL Game Predictions',
  description: 'AI-powered NFL game predictions with confidence scores',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}

