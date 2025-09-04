import { notFound } from 'next/navigation'
import Header from '@/components/Header'
import Footer from '@/components/Footer'
import ChapterContent from '@/components/ChapterContent'
import { getChapterBySlug } from '@/data/chapters'

interface PageProps {
  params: {
    slug: string
  }
}

export default function ChapterPage({ params }: PageProps) {
  const chapter = getChapterBySlug(params.slug)

  if (!chapter) {
    notFound()
  }

  return (
    <>
      <Header />
      <main>
        <ChapterContent chapter={chapter} />
      </main>
      <Footer />
    </>
  )
}

export async function generateStaticParams() {
  // This would typically fetch from your data source
  const chapters = [
    'prompt-chaining',
    'routing', 
    'parallelization',
    'reflection',
    'tool-use',
    'planning',
    'multi-agent',
    'memory-management',
    'learning-adaptation',
    'model-context-protocol',
    'goal-setting-monitoring',
    'exception-handling',
    'human-in-loop',
    'knowledge-retrieval',
    'inter-agent-communication',
    'resource-aware-optimization',
    'reasoning-techniques',
    'guardrails-safety',
    'evaluation-monitoring',
    'prioritization',
    'exploration-discovery'
  ]

  return chapters.map((slug) => ({
    slug: slug,
  }))
}
