import Header from '@/components/Header'
import Footer from '@/components/Footer'
import { getAllChapters } from '@/data/chapters'
import Link from 'next/link'
import { Clock, Code, ArrowRight } from 'lucide-react'

export default function PatternsPage() {
  const chapters = getAllChapters()

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800'
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800'
      case 'advanced':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <>
      <Header />
      <main className="section">
        <div className="container">
          <div className="text-center mb-12">
            <h1 className="text-4xl lg:text-5xl font-bold mb-4">
              All Chapters
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Browse all {chapters.length} chapters in our comprehensive guide to agentic patterns,
              organized by difficulty and topic area.
            </p>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-6">
            {chapters.map((chapter) => (
              <Link
                key={chapter.id}
                href={`/chapters/${chapter.id}`}
                className="group bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all duration-200 hover:border-primary-300"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="w-10 h-10 bg-primary-100 text-primary-600 rounded-lg flex items-center justify-center font-semibold">
                      {chapter.number}
                    </span>
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                        {chapter.title}
                      </h3>
                      <p className="text-sm text-gray-500">{chapter.part}</p>
                    </div>
                  </div>
                  <ArrowRight className="text-gray-400 group-hover:text-primary-600 transition-colors" size={20} />
                </div>
                
                <p className="text-gray-600 mb-4 line-clamp-2">
                  {chapter.description}
                </p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <Clock size={16} className="text-gray-400" />
                      <span className="text-gray-500">{chapter.readingTime}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Code size={16} className="text-gray-400" />
                      <span className="text-gray-500">Code included</span>
                    </div>
                  </div>
                  {chapter.difficulty && (
                    <span className={`text-xs px-3 py-1 rounded-full ${getDifficultyColor(chapter.difficulty)}`}>
                      {chapter.difficulty}
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>

          <div className="mt-12 text-center">
            <Link href="/table-of-contents" className="btn-secondary inline-flex items-center">
              View Full Table of Contents
              <ArrowRight className="ml-2" size={16} />
            </Link>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}
