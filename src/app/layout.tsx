import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Agentic Patterns - Understanding AI Agent Architectures',
  description: 'Explore comprehensive patterns and best practices for building autonomous AI agents and agentic systems.',
  keywords: 'agentic patterns, AI agents, autonomous systems, artificial intelligence, agent architectures',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen bg-gray-50">
          {children}
        </div>
      </body>
    </html>
  )
}
