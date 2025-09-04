import Header from '@/components/Header'
import Hero from '@/components/Hero'
import TableOfContents from '@/components/TableOfContents'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <>
      <Header />
      <main>
        <Hero />
        <TableOfContents />
      </main>
      <Footer />
    </>
  )
}
