import { notFound } from 'next/navigation'
import Header from '@/components/Header'
import Footer from '@/components/Footer'
import AppendixContent from '@/components/AppendixContent'
import { getAppendixBySlug } from '@/data/appendices'

interface PageProps {
  params: {
    slug: string
  }
}

export default function AppendixPage({ params }: PageProps) {
  const appendix = getAppendixBySlug(params.slug)

  if (!appendix) {
    notFound()
  }

  return (
    <>
      <Header />
      <main>
        <AppendixContent appendix={appendix} />
      </main>
      <Footer />
    </>
  )
}
