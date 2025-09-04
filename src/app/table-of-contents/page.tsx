import Header from '@/components/Header'
import Footer from '@/components/Footer'
import TableOfContents from '@/components/TableOfContents'

export const metadata = {
  title: 'Table of Contents - Agentic Patterns',
  description: 'Complete guide to agentic patterns with 21 chapters covering foundations, learning, human-centric patterns, and scaling.',
}

export default function TableOfContentsPage() {
  return (
    <>
      <Header />
      <main>
        <TableOfContents />
      </main>
      <Footer />
    </>
  )
}
